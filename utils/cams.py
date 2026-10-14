import torch
import numpy as np
import cv2
import random
from PIL import Image
from torchvision.transforms import GaussianBlur
import time
def generate_cam_by_features(features, target_class, model=None,dataset='cifar'):
    if dataset == 'cifar':
        size = 32
    elif dataset == 'clothing1m':
        size = 224
    elif dataset == 'animal10n':
        size = 64
    all_weights = model.linear.weight  # [num_classes, num_features]
    target_weights = all_weights[target_class]  # target-class weights for the batch [batch_size, num_features]
    if len(target_weights.shape) == 1:
        target_weights = target_weights.reshape(1, -1)  # add batch dimension
    # vectorized computation
    batch_features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
    batch_cams = torch.einsum('bhwc,bc->bhw',
                              batch_features,
                              target_weights)
    batch_cams = torch.relu(batch_cams)
    # get batch size and current spatial size
    batch_size, h, w = batch_cams.shape
    # resize with torch.nn.functional.interpolate (a channel dim is required)
    batch_cams = batch_cams.view(batch_size, 1, h, w)  # add channel dimension
    batch_cams = torch.nn.functional.interpolate(
        batch_cams,
        size=(size, size),
        mode='bilinear',
        align_corners=False
    )
    batch_cams = batch_cams.view(batch_size, size, size)  # remove channel dimension
    # normalize and scale to 0-255 (computed on GPU)
    # per-sample min and max
    batch_min = torch.min(batch_cams.view(batch_size, -1), dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    batch_max = torch.max(batch_cams.view(batch_size, -1), dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    # avoid division by zero
    epsilon = 1e-12
    batch_cams = 255 * (batch_cams - batch_min) / (batch_max - batch_min + epsilon)
    # cast to uint8
    batch_cams = batch_cams.to(dtype=torch.uint8)

    return batch_cams


def generate_cam_by_features_vgg1(features, target_class, model=None, dataset='animal10n'):
    size = 64
    # weights of the first FC layer of the VGG network
    # first FC layer weight shape: [4096, 512*7*7]
    first_fc_weight = model.linear[0].weight

    # weights of the last FC layer, shape [num_classes, 4096]
    last_fc_weight = model.linear[6].weight

    # direct weight mapping from feature maps to class scores
    # combine the two FC layers' weights with matrix multiplication
    # result shape: [num_classes, 512*7*7]
    combined_weights = torch.mm(last_fc_weight, first_fc_weight)

    # combined weights for the target classes
    target_weights = combined_weights[target_class]  # [batch_size, 512*7*7]
    if len(target_weights.shape) == 1:
        target_weights = target_weights.reshape(1, -1)  # add batch dimension

    # reshape weights to match the feature map's spatial structure
    # from [batch_size, 512*7*7] to [batch_size, 512, 7, 7]
    # then downsample 7x7 to the feature map's spatial size
    batch_size = target_weights.size(0)
    spatial_size = features.size(2)  # feature map height and width

    # reshape weights
    weights_reshaped = target_weights.reshape(batch_size, 512, 4, 4)

    # resize weights spatially to match the feature map
    weights_reshaped = torch.nn.functional.interpolate(
        weights_reshaped,
        size=(spatial_size, spatial_size),
        mode='bilinear',
        align_corners=False
    )

    # vectorized computation: weighted combination of feature maps and weights
    batch_features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
    weights_permuted = weights_reshaped.permute(0, 2, 3, 1)  # [B, H, W, C]

    # compute CAM via element-wise multiplication and summation
    batch_cams = torch.sum(batch_features * weights_permuted, dim=3)

    batch_cams = torch.relu(batch_cams)
    # get batch size and current spatial size
    batch_size, h, w = batch_cams.shape
    # resize with torch.nn.functional.interpolate (a channel dim is required)
    batch_cams = batch_cams.view(batch_size, 1, h, w)  # add channel dimension
    batch_cams = torch.nn.functional.interpolate(
        batch_cams,
        size=(size, size),
        mode='bilinear',
        align_corners=False
    )
    batch_cams = batch_cams.view(batch_size, size, size)  # remove channel dimension
    # normalize and scale to 0-255 (computed on GPU)
    # per-sample min and max
    batch_min = torch.min(batch_cams.view(batch_size, -1), dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    batch_max = torch.max(batch_cams.view(batch_size, -1), dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    # avoid division by zero
    epsilon = 1e-12
    batch_cams = 255 * (batch_cams - batch_min) / (batch_max - batch_min + epsilon)
    # cast to uint8
    batch_cams = batch_cams.to(dtype=torch.uint8)
    return batch_cams

def generate_cam_by_features_vgg(features, target_class, model=None,dataset='animal10n'):

    size = 64

    # weights of the last FC layer [10, 4096]
    last_fc_weight = model.linear.weight
    # weights of the second FC layer [4096, 4096]
    second_fc_weight = model.classifier[3].weight
    # weights of the first FC layer [4096, 25088]
    first_fc_weight = model.classifier[0].weight
    # target-class weights for the batch [batch_size, 4096]
    target_weights = last_fc_weight[target_class]
    # add batch dimension if there is only one sample
    if len(target_weights.shape) == 1:
        target_weights = target_weights.reshape(1, -1)
    # vectorized computation: per-sample weight combination
    # correct weight combination order: last_fc_weight -> second_fc_weight -> first_fc_weight
    # [batch_size, 4096] x [4096, 4096] = [batch_size, 4096]
    combined_weights_23 = torch.matmul(target_weights, second_fc_weight)
    # [batch_size, 4096] x [4096, 25088] = [batch_size, 25088]
    combined_weights = torch.matmul(combined_weights_23, first_fc_weight)
    # reshape weights to match the feature map's spatial structure [batch_size, 25088] -> [batch_size, 512, 7, 7]
    batch_size = features.size(0)
    weights_reshaped = combined_weights.reshape(batch_size, 512, 7, 7)
    # resize weights spatially to match the feature map
    spatial_size = features.size(2)  # feature map height and width
    weights_reshaped = torch.nn.functional.interpolate(
        weights_reshaped,
        size=(spatial_size, spatial_size),
        mode='bilinear',
        align_corners=False
    )
    # compute CAM via element-wise multiplication and summation
    # [B, 512, H, W] * [B, 512, H, W] -> [B, H, W]
    batch_cams = torch.sum(features * weights_reshaped, dim=1)
    # apply ReLU to keep only positive values
    batch_cams = torch.relu(batch_cams)
    # get batch size and current spatial size
    batch_size, h, w = batch_cams.shape
    # resize with torch.nn.functional.interpolate (a channel dim is required)
    batch_cams = batch_cams.view(batch_size, 1, h, w)  # add channel dimension
    batch_cams = torch.nn.functional.interpolate(
        batch_cams,
        size=(size, size),
        mode='bilinear',
        align_corners=False
    )
    batch_cams = batch_cams.view(batch_size, size, size)  # remove channel dimension
    # normalize and scale to 0-255 (computed on GPU)
    # per-sample min and max
    batch_min = torch.min(batch_cams.view(batch_size, -1), dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    batch_max = torch.max(batch_cams.view(batch_size, -1), dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    # avoid division by zero
    epsilon = 1e-12
    batch_cams = 255 * (batch_cams - batch_min) / (batch_max - batch_min + epsilon)
    # cast to uint8
    batch_cams = batch_cams.to(dtype=torch.uint8)
    return batch_cams


class CamExtractorVGG:
    """
    Extract feature maps from a VGG model, with support for selecting a target layer by name
    """

    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
        Forward through the conv layers, registering a hook on the specified layer
        """
        conv_output = None

        # name of the last conv layer
        last_conv_layer = None
        for module_name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module_name

        # iterate over all modules of the feature extractor
        for module_name, module in self.model.features._modules.items():
            x = module(x)  # forward

            # if a layer name is given, check for a match
            if self.target_layer_name and module_name == self.target_layer_name:
                # register the gradient hook directly, without detach()
                x.register_hook(self.save_gradient)
                conv_output = x  # save this layer's conv output
                break

            # if no layer name is given, use the last conv layer
            elif self.target_layer_name is None and module_name == last_conv_layer:
                # register the gradient hook directly, without detach()
                x.register_hook(self.save_gradient)
                conv_output = x  # save this layer's conv output
                break

        return conv_output, x

    def forward_pass(self, x):
        """
        Full forward pass on the model, returning the output of the last linear layer
        """
        # forward through the conv layers
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)  # flatten
        # forward through the classifier
        # use the last linear layer (self.linear) as the target output
        x = self.model.classifier(x)
        last_layer_output = self.model.linear(x)
        return conv_output, last_layer_output


def generate_grad_cam_vgg(features=None, target_class=None, model=None, dataset='animal10n', input_tensor=None,
                          target_layer_name=None):
    """
    Generate activation maps with Grad-CAM, with support for selecting a target layer by name:
    1. Select a layer by name (e.g. '35') or fall back to the last conv layer
    2. Compute gradients from the last linear layer (self.linear)
    3. More accurately reflect the regions the model attends to for a given class
    4. Fix gradient issues during training

    Args:
        target_layer_name: name of the target layer (e.g. '35'); if None, the last conv layer is used
    """
    size = 64

    # save the original model state
    original_training = model.training

    # switch to eval mode while keeping gradient computation
    model.eval()

    # make sure the input tensor requires grad
    if not input_tensor.requires_grad:
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

    # feature extractor with target layer selection by name
    extractor = CamExtractorVGG(model, target_layer_name)

    # full forward pass
    # conv_output is the conv output of the specified layer
    # last_layer_output is the output of the last linear layer
    conv_output, last_layer_output = extractor.forward_pass(input_tensor)

    # if no target class is given, use the top predicted class
    if target_class is None:
        target_class = torch.argmax(last_layer_output, dim=1).item()

    # one-hot output for backpropagation (based on the last linear layer)
    batch_size = last_layer_output.size(0)
    one_hot_output = torch.zeros(batch_size, last_layer_output.size(-1), device=last_layer_output.device)
    for i in range(batch_size):
        one_hot_output[i][target_class] = 1

    # backpropagate with the specified target (based on the last linear layer)
    # create_graph=False to avoid retaining the computation graph
    gradients = torch.autograd.grad(
        last_layer_output,
        conv_output,
        grad_outputs=one_hot_output,
        create_graph=False,
        retain_graph=False
    )[0]

    # weights from gradients (propagated from the last linear layer)
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

    # Grad-CAM formula: weighted combination of feature maps and gradient weights
    batch_cams = torch.sum(conv_output * weights, dim=1)  # [B, H, W]

    # apply ReLU to keep only positive values
    batch_cams = torch.relu(batch_cams)

    # get batch size and current spatial size
    batch_size, h, w = batch_cams.shape

    # resize with torch.nn.functional.interpolate
    batch_cams = batch_cams.view(batch_size, 1, h, w)
    batch_cams = torch.nn.functional.interpolate(
        batch_cams,
        size=(size, size),
        mode='bilinear',
        align_corners=False
    )
    batch_cams = batch_cams.view(batch_size, size, size)

    # normalize and scale to 0-255, keeping float dtype
    batch_min = torch.min(batch_cams.view(batch_size, -1), dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    batch_max = torch.max(batch_cams.view(batch_size, -1), dim=1, keepdim=True)[0].view(batch_size, 1, 1)

    epsilon = 1e-12
    batch_cams = 255 * (batch_cams - batch_min) / (batch_max - batch_min + epsilon)

    # detach before returning to avoid gradient issues
    batch_cams = batch_cams.detach()

    # restore the original model state
    if original_training:
        model.train()
    else:
        model.eval()

    return batch_cams


def apply_soft_mask(image, mask, blur_strength=0.8, blur_kernel=(5, 5), sigma=10):
    blurred = GaussianBlur(blur_kernel, sigma)(image.unsqueeze(0)).squeeze(0)
    mask_expanded = mask.unsqueeze(0).repeat(image.size(0), 1, 1).float()
    result = image * (1 - blur_strength * mask_expanded) + blurred * (blur_strength * mask_expanded)

    return result


def cam_mask(net, inputs, features, targets, prob, pred, ratio=0.95,xi=0.9, batch_size=64, dataset='cifar'):
    if dataset == 'cifar':
        size = 32
        threshold_max = 0.95
        threshold_min = 0.9
    elif dataset == 'clothing1m':
        size = 224
    elif dataset == 'animal10n':
        size = 64
        threshold_max = 0.95
        threshold_min = 0.9
    predicted = torch.argmax(pred, dim=1)
    if dataset == 'animal10n':
        cams_targets = generate_cam_by_features_vgg(features, targets, model=net, dataset=dataset)
        cams_preds = generate_cam_by_features_vgg(features, predicted, model=net, dataset=dataset)
    else:
        cams_targets = generate_cam_by_features(features, targets, model=net, dataset=dataset)
        cams_preds = generate_cam_by_features(features, predicted, model=net, dataset=dataset)
    device = inputs.device
    images_targets = inputs.clone()  # [clean, clean, noisy, noisy]
    images_preds = inputs.clone()
    images_front = inputs.clone()
    images_back = inputs.clone()
    front_masks = []
    for i in range(inputs.size(0)):
        cam_target = cams_targets[i]
        cam_pred = cams_preds[i]
        if i >= batch_size * 2:  # noisy
            random_factor = torch.rand(1).item() * (1-xi) + xi
            threshold_pred = cam_pred.max() * (1 - prob[i]) * ratio * random_factor
            if dataset == 'animal10n':
                random_factor = torch.rand(1).item() * (1-xi) + xi
                threshold_target = cam_target.max() * max(min(1 - prob[i], threshold_max), threshold_min) * random_factor
                part_targets = (cam_target > threshold_target)
            else:
                threshold_target = cam_target.float().mean()
                part_targets = (cam_target > threshold_target)  # for noisy samples, mask most of the region activated by the given label; other datasets need extra care
            part_pred = (cam_pred > threshold_pred)  # for noisy samples, mask the small region activated by the predicted label

            mask_threshold = cam_pred.float().mean()
            noisy_front = (cam_pred > mask_threshold)
            noisy_back = (cam_pred < mask_threshold)
            images_front[i, :, noisy_back] = 0
            images_back[i, :, noisy_front] = 0
            front_masks.append(noisy_front)

            part_targets_num = part_targets.sum()
            if part_targets_num == 0:
                # define the four corner regions
                corners = [
                    (0, 0, int(size / 4), int(size / 4)),  # top-left (r_start, c_start, r_end, c_end)
                    (0, int(size * 0.75), int(size / 4), size),  # top-right
                    (int(size * 0.75), 0, size, int(size / 4)),  # bottom-left
                    (int(size * 0.75), int(size * 0.75), size, size)  # bottom-right
                ]
                chosen_corner = random.choice(corners)
                r_start, c_start, r_end, c_end = chosen_corner
                part_targets[r_start:r_end, c_start:c_end] = True

            images_targets[i, :, part_targets] = 0

            images_preds[i, :, part_pred] = 0


        else:  # clean
            random_factor = torch.rand(1).item() * (1-xi) + xi
            threshold_pred = cam_pred.max() * (prob[i]) * ratio * random_factor
            threshold_target = cam_target.float().mean()
            part_targets = (cam_target < threshold_target)  # for clean samples, mask the background
            part_pred = (cam_pred > threshold_pred)  # for clean samples, mask the small region activated by the predicted label

            mask_threshold = cam_pred.float().mean()
            clean_front = (cam_pred > mask_threshold)
            clean_back = (cam_pred < mask_threshold)
            images_front[i, :, clean_back] = 0
            images_back[i, :, clean_front] = 0

            front_masks.append(clean_front)

            images_targets[i, :, part_targets] = 0

            images_preds[i, :, part_pred] = 0


    # combine images
    idx = torch.randperm(inputs.size(0))
    spatial_mask = (images_front.any(dim=1) > 0)
    mask = spatial_mask.unsqueeze(1).expand_as(images_front)
    modified_back = images_back[idx]
    modified_back[mask] = 0
    back_zero_mask = (modified_back == 0)
    modified_back[back_zero_mask] = inputs[back_zero_mask]
    combined_images = modified_back

    clean_mask = torch.cat((images_targets[:batch_size * 2], images_preds[:batch_size * 2]), dim=0)
    noisy_mask = torch.cat((images_targets[batch_size * 2:], images_preds[batch_size * 2:]), dim=0)

    return clean_mask, noisy_mask, combined_images, front_masks
