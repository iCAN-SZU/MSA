import argparse
import random
import sys
import time

import torch.backends.cudnn as cudnn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import CosineAnnealingLR

import dataloader_cifar as dataloader
from PreResNet import *
from losses.loss_EDL import EDL_Loss
from utils.cams import generate_cam_by_features, cam_mask
from utils.data import *
from utils.tools import D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument('--noise_mode', default='sym')
    parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--r', default=0.8, type=float, help='noise ratio')
    parser.add_argument('--id', default='')
    parser.add_argument('--seed', default=123)
    parser.add_argument('--loss', default='ssim', type=str)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--data_path', default='data/cifar10', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--MAX_HISTORY', default=20, type=int)
    parser.add_argument('--resume1', default='', type=str, help='path to checkpoint')
    parser.add_argument('--resume2', default='', type=str, help='path to checkpoint')
    parser.add_argument('--fc', default=False, type=bool)
    parser.add_argument('--ratio', default=0.6, type=float)
    parser.add_argument('--xi', default=0.9, type=float)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    # Training
    def train(epoch, net, net2, optimizer, scheduler, labeled_trainloader, unlabeled_trainloader):
        net.train()
        net2.eval()  # fix one network and train the other
        # track accuracy on the training set
        total_correct_x = 0
        total_correct_xt = 0
        total_samples_x = 0
        total_correct_u = 0
        total_correct_ut = 0
        total_samples_u = 0
        w_lu = len(unlabeled_trainloader.dataset) / 50000

        unlabeled_train_iter = iter(unlabeled_trainloader)
        num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
        for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, true_x) in enumerate(labeled_trainloader):
            try:
                inputs_u, inputs_u2, labels_u, w_u, true_u = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, labels_u, w_u, true_u = next(unlabeled_train_iter)
            batch_size = inputs_x.size(0)
            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
            w_x = w_x.view(-1, 1).type(torch.FloatTensor)
            w_u = w_u.view(-1, 1).type(torch.FloatTensor)
            inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2, labels_u, w_u = inputs_u.cuda(), inputs_u2.cuda(), labels_u.cuda(), w_u.cuda()
            true_x, true_u = true_x.cuda(), true_u.cuda()

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                features_u, outputs_u11, _ = net(inputs_u)
                features_u2, outputs_u12, _ = net(inputs_u2)
                _, outputs_u21, _ = net2(inputs_u)
                _, outputs_u22, _ = net2(inputs_u2)

                # label refinement of labeled samples
                features_x, outputs_x11_h1, _ = net(inputs_x)
                features_x2, outputs_x12_h1, _ = net(inputs_x2)
                _, outputs_x21, _ = net2(inputs_x)
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u21, dim=1)) / 2

                px = (torch.softmax(outputs_x11_h1, dim=1) + torch.softmax(outputs_x21, dim=1)) / 2

                mask_pred = torch.cat([px, px, pu, pu], dim=0)
                mask_w = torch.cat([w_x, w_x, w_u, w_u], dim=0)
                mask_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
                mask_features = torch.cat([features_x, features_x2, features_u, features_u2], dim=0)
                mask_targets = torch.cat([labels_x.argmax(1), labels_x.argmax(1), labels_u, labels_u], dim=0)
                clean_mask, noisy_mask, combined_images, front_masks = cam_mask(net, mask_inputs,
                                                                                              mask_features,
                                                                                              mask_targets, mask_w,
                                                                                              mask_pred,
                                                                                              ratio=args.ratio,
                                                                                              xi=args.xi,
                                                                                              batch_size=batch_size)
                ptu = pu ** (1 / args.T)  # temparature sharpening
                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1 / args.T)  # temparature sharpening
                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()
                u_outputs = pu
                x_outputs = px

            fc = args.fc
            if fc:
                weak_clean_mask = clean_mask[batch_size * 2:batch_size * 3]
                weak_noisy_mask = noisy_mask[batch_size * 2:batch_size * 3]
                strong_clean_mask = clean_mask[batch_size * 3:]
                strong_noisy_mask = noisy_mask[batch_size * 3:]
                fc_inputs = torch.cat([inputs_x, inputs_u, inputs_x2, inputs_u2,
                                       weak_clean_mask, weak_noisy_mask, strong_clean_mask, strong_noisy_mask], dim=0)

                f_proj, f_pred = net(fc_inputs, fc=True)
                Lfc1 = D(f_pred[batch_size * 4:], f_proj[:batch_size * 4])
                Lfc2 = D(f_pred[batch_size * 2:batch_size * 4], f_proj[:batch_size * 2])
                Lfc = Lfc1 + Lfc2

            # compute prediction accuracy on labeled and unlabeled data
            with torch.no_grad():
                # labeled sample accuracy
                _, pred_x = torch.max(x_outputs.detach(), 1)
                correct_xt = pred_x.eq(true_x)
                correct_x = pred_x.eq(torch.max(labels_x, 1)[1])
                total_correct_x += correct_x.sum().item()
                total_correct_xt += correct_xt.sum().item()
                total_samples_x += correct_x.size(0)

                # unlabeled sample accuracy
                _, pred_u = torch.max(u_outputs.detach(), 1)
                correct_ut = pred_u.eq(true_u)
                correct_u = pred_u.eq(labels_u)
                total_correct_u += correct_u.sum().item()
                total_correct_ut += correct_ut.sum().item()
                total_samples_u += correct_u.size(0)

            # mixmatch
            w = linear_rampup(epoch + batch_idx / num_iter, warm_up, 30)
            w2 = linear_rampup(epoch + batch_idx / num_iter, warm_up + 20, 30)
            if args.r == 0.9:
                w = linear_rampup(epoch + batch_idx / num_iter, warm_up, 30)
                w2 = linear_rampup(epoch + batch_idx / num_iter, warm_up + 20, 30)
            # labeled samples

            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)
            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
            idx_all = torch.randperm(all_inputs.size(0))
            all_inputs_a, all_inputs_b = all_inputs, all_inputs[idx_all]
            all_targets_a, all_targets_b = all_targets, all_targets[idx_all]
            mixed_inputs = l * all_inputs_a + (1 - l) * all_inputs_b
            mixed_targets = l * all_targets_a + (1 - l) * all_targets_b

            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)
            clean_inputs = torch.cat([inputs_x, inputs_x2], dim=0)
            clean_targets = torch.cat([targets_x, targets_x], dim=0)
            idx_clean = torch.randperm(clean_inputs.size(0))
            clean_inputs_a, clean_inputs_b = clean_inputs, clean_inputs[idx_clean]
            clean_targets_a, clean_targets_b = clean_targets, clean_targets[idx_clean]
            mixed_clean_inputs = l * clean_inputs_a + (1 - l) * clean_inputs_b
            mixed_clean_targets = l * clean_targets_a + (1 - l) * clean_targets_b

            comb = True  # whether to use combined images
            use_mask = True

            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)
            if comb:
                clean_masked_inputs = clean_mask
                clean_masked_targets = torch.cat([targets_x, targets_x, targets_x, targets_x], dim=0)
            else:
                clean_masked_inputs = clean_mask
                clean_masked_targets = torch.cat([targets_x, targets_x, targets_x, targets_x], dim=0)
            idx_clean_mask = torch.randperm(clean_masked_inputs.size(0))
            clean_masked_inputs_a, clean_masked_inputs_b = clean_masked_inputs, clean_masked_inputs[idx_clean_mask]
            clean_masked_targets_a, clean_masked_targets_b = clean_masked_targets, clean_masked_targets[idx_clean_mask]
            mixed_clean_masked_inputs = l * clean_masked_inputs_a + (1 - l) * clean_masked_inputs_b
            mixed_clean_masked_targets = l * clean_masked_targets_a + (1 - l) * clean_masked_targets_b

            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)
            if comb:
                masked_inputs = torch.cat(
                    [clean_mask, combined_images[:batch_size * 2], noisy_mask, combined_images[batch_size * 2:]], dim=0)
                masked_targets = torch.cat([targets_x, targets_x, targets_x, targets_x, targets_x, targets_x,
                                            targets_u, targets_u, targets_u, targets_u, targets_u, targets_u], dim=0)
            else:
                masked_inputs = torch.cat(
                    [clean_mask, noisy_mask], dim=0)
                masked_targets = torch.cat([targets_x, targets_x, targets_x, targets_x,
                                            targets_u, targets_u, targets_u, targets_u], dim=0)
            idx_mask = torch.randperm(masked_inputs.size(0))
            mask_inputs_a, mask_inputs_b = masked_inputs, masked_inputs[idx_mask]
            mask_targets_a, mask_targets_b = masked_targets, masked_targets[idx_mask]
            mixed_mask_inputs = l * mask_inputs_a + (1 - l) * mask_inputs_b
            mixed_mask_targets = l * mask_targets_a + (1 - l) * mask_targets_b

            _, mixed_clean_outputs1, _ = net(mixed_clean_inputs)
            _, mixed_outputs1, mixed_outputs2 = net(mixed_inputs)
            if use_mask:
                _, mixed_clean_masked_outputs1, mixed_clean_masked_outputs2 = net(mixed_clean_masked_inputs)
                _, mask_outputs1, mask_outputs2 = net(mixed_mask_inputs)

            Lx11 = EDL(mixed_clean_outputs1, mixed_clean_targets)
            Lx21 = EDL(mixed_outputs2[:batch_size * 2], mixed_targets[:batch_size * 2])

            if use_mask:
                if comb:
                    Lx12 = EDL(mixed_clean_masked_outputs1, mixed_clean_masked_targets)
                    Lx22 = EDL(mask_outputs2[:batch_size * 6], mixed_mask_targets[:batch_size * 6])
                else:
                    Lx12 = EDL(mixed_clean_masked_outputs1, mixed_clean_masked_targets)
                    Lx22 = EDL(mask_outputs2[:batch_size * 4], mixed_mask_targets[:batch_size * 4])
            if use_mask:
                Lx = Lx11 + Lx21 + w * (Lx12 + Lx22)
            else:
                Lx = Lx11 + Lx21

            # unsupervised
            Lu1 = EDL(mixed_outputs2[batch_size * 2:], mixed_targets[batch_size * 2:])
            probs_u = torch.softmax(mixed_outputs2[batch_size * 2:], dim=1)

            Lu2 = torch.mean((probs_u - mixed_targets[batch_size * 2:]) ** 2)
            if use_mask:
                if comb:
                    probs_u_mask = torch.softmax(mask_outputs2[batch_size * 6:], dim=1)
                    Lu2_mask = torch.mean((probs_u_mask - mixed_mask_targets[batch_size * 6:]) ** 2)
                    Lu1_mask = EDL(mask_outputs2[batch_size * 6:], mixed_mask_targets[batch_size * 6:])
                else:
                    probs_u_mask = torch.softmax(mask_outputs2[batch_size * 4:], dim=1)
                    Lu2_mask = torch.mean((probs_u_mask - mixed_mask_targets[batch_size * 4:]) ** 2)
                    Lu1_mask = EDL(mask_outputs2[batch_size * 4:], mixed_mask_targets[batch_size * 4:])

            if use_mask:
                Lu_ce = w2 * Lu1 + w2 * Lu1_mask
                Lu_mse = args.lambda_u * w * Lu2 + args.lambda_u * w * Lu2_mask
            else:
                Lu_ce = w2 * Lu1
                Lu_mse = args.lambda_u * w * Lu2

            Lu = Lu_mse + w_lu * Lu_ce

            # regularization
            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.cuda()
            pred_mean1 = torch.softmax(mixed_clean_outputs1, dim=1).mean(0)

            penalty1 = torch.sum(prior * torch.log(prior / pred_mean1))
            penalty = penalty1

            if fc :
                loss = Lx + Lu + penalty + (Lfc)
            else:
                loss = Lx + Lu + penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sys.stdout.write('\r')
            sys.stdout.write(
                '%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t lr: %3f Labeled loss: %.2f  Unlabeled loss: %.2f '
                % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                   optimizer.param_groups[0]['lr'],
                   Lx.item(), Lu.item()))
            if fc:
                sys.stdout.write('Lfc: %.2f' % Lfc)
            sys.stdout.flush()
        scheduler.step()
        epoch_acc_x = total_correct_x / total_samples_x
        epoch_acc_u = total_correct_u / total_samples_u
        epoch_acc_xt = total_correct_xt / total_samples_x
        epoch_acc_ut = total_correct_ut / total_samples_u

        # print epoch-level accuracy statistics
        print_msg = f' Epoch {epoch} - X_acc: {epoch_acc_x:.3f} U_acc: {epoch_acc_u:.3f} X_true: {epoch_acc_xt:.3f} U_true: {epoch_acc_ut:.3f} w_lu:{w_lu:.3f}'
        print(print_msg)


    def warmup(epoch, net, optimizer, dataloader, all_area, all_dist, cams_bank, re=False):
        area = torch.zeros(50000, device='cuda')
        dist = torch.zeros(50000, device='cuda')
        epoch_cams = torch.zeros((50000, 32, 32), dtype=torch.uint8, device='cuda')

        net.train()
        num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
        for batch_idx, (inputs, labels, index, true_label) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            features, outputs1, outputs2 = net(inputs)
            cams1 = generate_cam_by_features(features=features, target_class=labels, model=net)
            labels = torch.zeros(inputs.size(0), args.num_class, device='cuda').scatter_(1, labels.view(-1, 1), 1)
            loss = EDL_Loss(args.num_class)(outputs1, labels)

            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(outputs1, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))
            L = loss + penalty
            if re == False:
                L.backward()
                optimizer.step()
            else:
                pass
            epoch_cams[index] = cams1

            if epoch > (warm_up / 2):
                area[index] = calc_core_size(cams1, size=0.1)
                dist[index] = calc_centroid_displacement(cams_bank[-1][index], cams1)

            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                             % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                                loss.item()))
            sys.stdout.flush()
        cams_bank.append(epoch_cams)
        cams_bank = cams_bank[-1:]
        all_area.append(area)
        all_dist.append(dist)

        return all_area, all_dist, cams_bank


    def test(epoch, net1, net2, best_acc):
        net1.eval()
        net2.eval()
        correct_ori = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                _, ori_outputs1, outputs1 = net1(inputs)
                _, ori_outputs2, outputs2 = net2(inputs)
                ori_outputs = ori_outputs1 + ori_outputs2
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)
                _, predicted_ori = torch.max(ori_outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
                correct_ori += predicted_ori.eq(targets).sum().item()
        acc = 100. * correct / total
        acc_ori = 100. * correct_ori / total
        # update best accuracy
        if acc > best_acc:
            best_acc = acc
        print("\n| Test Epoch #%d\t Accuracy: %.2f%% Head1 acc: %.2f%% Best Accuracy: %.2f%%\n" % (
            epoch, acc, acc_ori, best_acc))
        test_log.write(
            'Epoch:%d   Head2 acc:%.2f Head1 acc: %.2f%% Best Accuracy: %.2f%% lr:%.4f\n' % (
            epoch, acc, acc_ori, best_acc, optimizer1.param_groups[0]['lr']))
        test_log.flush()
        return acc, best_acc


    def eval_train(model, epoch, all_loss, all_mse_loss, all_area, all_dist, cams_bank,
                       use_cam=True,
                       use_margin=True):

        model.eval()
        losses = torch.zeros(50000, device='cuda')
        margins = torch.zeros(50000, device='cuda')
        area = torch.zeros(50000, device='cuda')
        dist = torch.zeros(50000, device='cuda')
        epoch_cams = torch.zeros((50000, 32, 32), dtype=torch.uint8, device='cuda')
        correct_noisy = 0
        correct_real = 0
        total = 0
        if args.r == 0.9:
            current = np.clip((epoch - (warm_up + 30)) / 10, 0.0, 1.0)
        else:
            current = np.clip((epoch - (warm_up + 20)) / 10, 0.0, 1.0)
        if epoch > (200):
            current = 0
        CAM = torch.zeros(50000)
        with torch.no_grad():
            for batch_idx, (inputs, targets, index, true_label) in enumerate(eval_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                true_label = true_label.cuda()
                features, outputs1, outputs2 = model(inputs)
                outputs = outputs1
                loss = CE(outputs, targets)
                _, predicted = torch.max(outputs, 1)

                if use_cam:
                    cams1 = generate_cam_by_features(features=features, target_class=targets, model=model)

                correct_noisy += predicted.eq(targets).sum().item()
                correct_real += predicted.eq(true_label).sum().item()
                total += targets.size(0)

                if use_margin:
                    batch_size = inputs.size(0)
                    batch_indices = torch.arange(batch_size, device=inputs.device)
                    evidence_pos = outputs[batch_indices, targets]
                    copy_outputs = outputs.clone()
                    copy_outputs[batch_indices, targets] = -1e5
                    evidence_neg, _ = copy_outputs.max(dim=1)
                    margins[index] = evidence_pos - evidence_neg

                losses[index] = loss
                if use_cam:
                    epoch_cams[index] = cams1
                    area[index] = calc_core_size(cams1, size=0.1)
                    dist[index] = calc_centroid_displacement(cams_bank[-1][index], cams1)

        if use_cam:
            area = (area - area.min()) / (area.max() - area.min() + 1)
            dist = (dist - dist.min()) / (dist.max() - dist.min() + 1)

            cams_bank.append(epoch_cams)
            cams_bank = cams_bank[-1:]
            all_area.append(area)
            all_dist.append(dist)
            HISTORY_WINDOW = args.MAX_HISTORY  # analyze history over the last n epochs

            area_stack = torch.stack(all_area[-HISTORY_WINDOW:])
            area_means = torch.mean(area_stack, dim=0)
            area_stds = torch.std(area_stack, dim=0)

            dist_stack = torch.stack(all_dist[-HISTORY_WINDOW:])
            dist_means = torch.mean(dist_stack, dim=0)
            dist_stds = torch.std(dist_stack, dim=0)

            CAM = (area_stds / (area_means + 1)) + dist_means + dist_stds
            CAM = (CAM - CAM.min()) / (CAM.max() - CAM.min())
            CAM = CAM * 10 / args.num_class
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        all_loss.append(losses)
        if args.r == 0.9:  # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(all_loss)
            losses = history[-5:].mean(0)

        if use_margin:
            margins = (margins - margins.min()) / (margins.max() - margins.min())

            if use_cam:
                if args.noise_mode == 'asym':
                    if args.dataset == 'cifar100':
                        margins = margins - current * CAM
                    else:

                        margins = current * margins - losses - current * CAM
                else:
                    if args.r == 0.9:
                        margins = margins - losses - current * CAM
                    else:
                        margins = margins - losses - current * CAM
            else:
                margins = margins - losses

            margins = (margins - margins.min()) / (margins.max() - margins.min())
            input = margins.reshape(-1, 1).cpu()
        else:
            input = losses.reshape(-1, 1).cpu()

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input)
        prob = gmm.predict_proba(input)
        if use_margin:
            prob = prob[:, gmm.means_.argmax()]
        else:
            prob = prob[:, gmm.means_.argmin()]

        pred = (prob > args.p_threshold)

        acc_n = 100. * correct_noisy / total  # overall accuracy
        acc_r = 100. * correct_real / total
        sys.stdout.write('Train Accuracy: %.2f%%  Train Real Accuracy: %.2f%% \n' % (acc_n, acc_r))
        return pred, prob, all_loss, all_mse_loss, all_area, all_dist, cams_bank


    def linear_rampup(current, warm_up, rampup_length=16):
        if current <= warm_up:
            factor = 0
        else:
            factor = (current - warm_up) / rampup_length

        return float(np.clip(factor, 0.0, 1.0))

    class NegEntropy(object):
        def __call__(self, outputs):
            probs = torch.softmax(outputs, dim=1)
            return torch.mean(torch.sum(probs.log() * probs, dim=1))


    def create_model(resume=''):
        if resume == '':
            model = ResNet18(num_classes=args.num_class)
        else:
            model = ResNet18(num_classes=args.num_class)
            checkpoint = torch.load(resume, weights_only=True)
            model.load_state_dict(checkpoint)
        return model.cuda()


    start_time = time.time()
    current_time = time.strftime("%Y%m%d_%H%M%S")
    file_suffix = '_fc' if args.fc else ''
    stats_log = open(
        './checkpoint/%s_%.1f_%s_%s_%s_%s' % (
        args.dataset, args.r, args.noise_mode, current_time, args.lambda_u, file_suffix) + '_stats.txt', 'w')
    test_log = open('./checkpoint/%s_%.1f_%s_%s_%s_%s' % (
    args.dataset, args.r, args.noise_mode, current_time, args.lambda_u, file_suffix) + '_acc.txt',
                    'w')

    if args.dataset == 'cifar10':
        warm_up = 10
        args.num_class = 10
    elif args.dataset == 'cifar100':
        warm_up = 30
        args.num_class = 100

    loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                         num_workers=5, \
                                         root_dir=args.data_path, log=stats_log,
                                         noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

    all_loss = [[], []]  # save the history of losses from two networks
    all_mse_loss = [[], []]  # history of CAM mse losses from the two networks
    cams_bank = [[], []]
    all_area = [[], []]
    all_dist = [[], []]
    best_acc = 0.0
    cudnn.benchmark = True

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    EDL = EDL_Loss(args.num_class)
    test_loader = loader.run('test')
    if args.noise_mode == 'asym':
        conf_penalty = NegEntropy()
    print('| Building net')
    if args.resume1 == '':
        net1 = create_model()
        net2 = create_model()
        optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        start_epoch = 0
    else:
        net1 = create_model(resume=args.resume1)
        net2 = create_model(resume=args.resume2)
        optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        start_epoch = 9
        warmup_trainloader = loader.run('warmup')
        all_area[0], all_dist[0], cams_bank[0] = warmup(9, net1, optimizer1, warmup_trainloader,
                                                        all_area[0], all_dist[0], cams_bank[0])
        all_area[1], all_dist[1], cams_bank[1] = warmup(9, net2, optimizer2, warmup_trainloader,
                                                        all_area[1], all_dist[1], cams_bank[1])
    cos = True
    if cos:
        scheduler1 = CosineAnnealingLR(optimizer1, T_max=args.num_epochs - warm_up, eta_min=0.0005)
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=args.num_epochs - warm_up, eta_min=0.0005)
    else:
        scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=140, gamma=0.1)
        scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=140, gamma=0.1)

    for epoch in range(start_epoch, args.num_epochs + 1):
        eval_loader = loader.run('eval_train')

        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net1')
            all_area[0], all_dist[0], cams_bank[0] = warmup(epoch, net1, optimizer1,
                                                            warmup_trainloader, all_area[0],
                                                            all_dist[0], cams_bank[0])
            print('\nWarmup Net2')
            all_area[1], all_dist[1], cams_bank[1] = warmup(epoch, net2, optimizer2,
                                                            warmup_trainloader, all_area[1],
                                                            all_dist[1], cams_bank[1])
        else:
            pred1_after, prob1, all_loss[0], all_mse_loss[0], all_area[0], all_dist[0], \
            cams_bank[0] = eval_train(net1, epoch, all_loss[0], all_mse_loss[0], all_area[0], all_dist[0],
                                          cams_bank[0])
            pred2_after, prob2, all_loss[1], all_mse_loss[1], all_area[1], all_dist[1], \
            cams_bank[1] = eval_train(net2, epoch, all_loss[1], all_mse_loss[1], all_area[1], all_dist[1],
                                          cams_bank[1])
            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2_after, prob2)  # co-divide
            train(epoch, net1, net2, optimizer1, scheduler1, labeled_trainloader, unlabeled_trainloader)  # train net1

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1_after, prob1)  # co-divide
            train(epoch, net2, net1, optimizer2, scheduler2, labeled_trainloader, unlabeled_trainloader)  # train net2

        acc, best_acc = test(epoch, net1, net2, best_acc)

    end_time = time.time() - start_time  # running time in seconds

    print("time：%d h %d m %d s" % (end_time // 3600, (end_time % 3600) // 60, end_time % 60))
