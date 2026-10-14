from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
from utils.autoaugment import CIFAR10Policy
from utils.Asymmetric_Noise import *
import torch.multiprocessing
from scipy import stats
from math import inf
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode='', noise_file='',
                 pred_after=[], probability=[], log=''):
        self.dataset = dataset
        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8}  # class transition for asymmetric noise

        mean_10 = (0.4914, 0.4822, 0.4465)
        std_10 = (0.2023, 0.1994, 0.2010)
        mean_100=(0.507, 0.487, 0.441)
        std_100= (0.267, 0.256, 0.276)
        if dataset=='cifar10':
           mean=mean_10
           std=std_10
        else:
           mean=mean_100
           std=std_100
        self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
        ])
        self.transform_strong = transforms.Compose([
           transforms.RandomCrop(32, padding=4),
           transforms.RandomHorizontalFlip(),
           CIFAR10Policy(),
           transforms.ToTensor(),
           transforms.Normalize(mean=mean, std=std)])

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            true_label = train_label
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))
            else:  # inject noise
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r * 50000)
                noise_idx = idx[:num_noise]
                if noise_mode == 'asym':
                    if dataset == 'cifar10':
                        noise_label, prob11 = noisify_cifar10_asymmetric(train_label, self.r)
                        noise_label = noise_label.tolist()
                    else:
                        noise_label, prob11 = noisify_cifar100_asymmetric(train_label, self.r)
                        noise_label = noise_label.tolist()
                elif noise_mode == 'sym':
                    for i in range(50000):
                        if i in noise_idx:
                            if dataset == 'cifar10':
                                noiselabel = random.randint(0, 9)
                            elif dataset == 'cifar100':
                                noiselabel = random.randint(0, 99)
                            noise_label.append(noiselabel)
                        else:
                            noise_label.append(train_label[i])
                elif noise_mode == 'ins':
                    noise_label = self.instance_noise(tau=self.r,train_data=train_data,true_label=true_label)
                print("save noisy labels to %s ..." % noise_file)
                json.dump(noise_label, open(noise_file, "w"))

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
                self.true_labels = true_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred_after.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    clean = (np.array(noise_label) == np.array(train_label))
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability, clean)
                    auc, _, _ = auc_meter.value()

                    # ratio of actually-clean samples among those predicted clean
                    clean_in_pred = np.sum(clean[pred_idx])  # samples predicted clean that are actually clean
                    total_pred = pred_after.sum()  # total samples predicted clean
                    clean_ratio = clean_in_pred / total_pred if total_pred > 0 else 0

                    # show per-class counts within the selected clean set
                    # distribution of true labels
                    print(f'选择样本数及干净比例：{total_pred}_{clean_ratio}')
                    noisy_labels = np.array(noise_label)[pred_idx]
                    true_labels = np.array(true_label)[pred_idx]
                    # distribution of true labels and noisy labels
                    class_counts_true = dict(zip(*np.unique(true_labels, return_counts=True)))
                    class_counts_noisy = dict(zip(*np.unique(noisy_labels, return_counts=True)))

                    # build visualization data with percentages
                    total = len(true_labels)
                    class_distribution = {
                        str(cls): {
                            "true_count": int(class_counts_true.get(cls, 0)),
                            "true_percentage": round(class_counts_true.get(cls, 0) / total * 100, 2),
                            "noisy_count": int(class_counts_noisy.get(cls, 0)),
                            "noisy_percentage": round(class_counts_noisy.get(cls, 0) / total * 100, 2)
                        } for cls in set(list(class_counts_true.keys()) + list(class_counts_noisy.keys()))
                    }

                    # save to a JSON file
                    stats_path = os.path.join('checkpoint', f'class_dist_{dataset}_{noise_mode}_{r}.json')
                    with open(stats_path, 'w') as f:
                        json.dump(class_distribution, f, indent=2)

                    log.write(
                        'labeled samples:%d  AUC:%.3f  Clean Ratio in Pred:%.3f  True Clean Number:%d  Clean in Pred:%d  Noise in Pred:%d \n'
                        % (pred_after.sum(), auc, clean_ratio, clean.sum(), clean_in_pred,
                           total_pred - clean_in_pred))
                    log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred_after).nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                self.train_data = train_data[pred_idx]
                self.true_labels = [true_label[i] for i in pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def instance_noise(self,train_data,true_label,tau=0.2, std=0.1):
        num_samples = len(train_data)
        num_classes = 10 if self.dataset == 'cifar10' else 100
        feature_size = 3 * 32 * 32  # feature size of a CIFAR image
        # initialize the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # sample flip rate q from truncated normal N(tau, sigma^2, [0, 1])
        flip_distribution = stats.truncnorm(
            (0 - tau) / std,  # lower bound
            (1 - tau) / std,  # upper bound
            loc=tau,  # mean
            scale=std  # std
        )
        q = flip_distribution.rvs(num_samples)
        # sample weight matrix W in R^{K x dim x K} from standard normal N(0, 1)
        W = torch.tensor(
            np.random.randn(num_classes, feature_size, num_classes)
        ).float().to(device)
        P = []  # transition matrix of each sample
        for i in range(num_samples):
            # load the raw image and transform it
            img = train_data[i]
            x = self.test_transform(Image.fromarray(img)).to(device)
            y = true_label[i]  # use the true label as y_i
            # instance-dependent flip rate
            # x(1x3072) x W[y](3072xK) = 1xK gives the p vector
            p = x.reshape(1, -1).mm(W[y]).squeeze(0)
            # control the diagonal element so that p[y] is 0 after softmax
            p[y] = -inf
            # make the off-diagonal elements sum to q_i
            p = q[i] * F.softmax(p, dim=0)
            p[y] += 1 - q[i]  # set the diagonal element to 1 - q_i
            P.append(p)
        # convert to a numpy array and generate new labels
        P = torch.stack(P, 0).cpu().numpy()
        class_list = list(range(num_classes))
        new_labels = [np.random.choice(class_list, p=P[i]) for i in range(num_samples)]
        # compute and print the noise rate
        noise_rate = (np.array(new_labels) != np.array(true_label)).mean()
        print(f'noise rate =  {noise_rate:.4f}')
        new_labels = [int(label) for label in new_labels]
        return new_labels

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob, true_label = self.train_data[index], self.noise_label[index], self.probability[index], self.true_labels[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_strong(img)
            return img1, img2, target, prob, true_label
        elif self.mode == 'unlabeled':
            img, target, prob, true_label = self.train_data[index], self.noise_label[index], self.probability[index], self.true_labels[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_strong(img)
            return img1, img2, target, prob, true_label
        elif self.mode == 'all':
            true_label = self.true_labels[index]
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index, true_label
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset == 'cifar10':
            self.num_classes = 10
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.transform_strong = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        elif self.dataset == 'cifar100':
            self.num_classes = 100
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
            ])
            self.transform_strong = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
            ])

    def run(self, mode, pred_after=[], prob=[]):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_train, mode="all",
                                        noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                            root_dir=self.root_dir, transform=self.transform_train, mode="labeled",
                                            noise_file=self.noise_file, pred_after=pred_after,
                                            probability=prob, log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                              root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",
                                              noise_file=self.noise_file, pred_after=pred_after, probability=prob)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                         noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
