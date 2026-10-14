import torch
import torch.nn as nn


class EDL_Loss(nn.Module):
    """
    evidence deep learning loss with label smoothing support
    """
    def __init__(self, num_classes):
        super(EDL_Loss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels):
        alpha = torch.exp(logits)+10./self.num_classes
        total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]

        # use smoothed (soft) labels directly; labels can be a probability distribution
        # labels.shape: [B, num_classes], each row is a probability distribution
        smooth_labels = labels
        smooth_labels.requires_grad = False
        loss_nll = torch.sum(smooth_labels * (total_alpha.log() - alpha.log()), dim=1)

        uniform_bata = torch.ones((1, logits.shape[1])).cuda()
        uniform_bata.requires_grad = False
        total_uniform_beta = torch.sum(uniform_bata, dim=1)

        # use smoothed labels instead of one-hot labels
        new_alpha = smooth_labels + (1.0 - smooth_labels) * (self.num_classes / 10.) * alpha
        new_total_alpha = torch.sum(new_alpha, dim=1)  # new_total_alpha.shape: [B]
        loss_kl = torch.lgamma(new_total_alpha) - torch.lgamma(total_uniform_beta) - torch.sum(torch.lgamma(new_alpha), dim=1) \
            + torch.sum((new_alpha - 1) * (torch.digamma(new_alpha) - torch.digamma(new_total_alpha.unsqueeze(1))), dim=1)
        loss_kl = 0.5 * loss_kl / self.num_classes

        return loss_nll.mean() + loss_kl.mean()
