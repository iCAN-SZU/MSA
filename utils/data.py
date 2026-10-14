import numpy as np
import torch

def calc_core_size(cams, size=0.5):
    batch_size = cams.size(0)
    results = torch.zeros(batch_size, device=cams.device)
    h, w = cams.shape[1], cams.shape[2]
    # find all-zero samples
    zero_mask = (cams.max(dim=1)[0].max(dim=1)[0] == 0)
    non_zero_mask = ~zero_mask

    # assign random values to all-zero samples
    zero_indices = torch.where(zero_mask)[0]
    if len(zero_indices) > 0:
        results[zero_indices] = torch.rand(len(zero_indices), device=cams.device) * 0.5 * (h * w)

    # compute the core region of non-zero samples
    if non_zero_mask.sum() > 0:
        results[non_zero_mask] = (cams[non_zero_mask] > 0).sum(dim=(1, 2)).float()

    return results


def calc_displacement(cam1, cam2, dataset='cifar'):
    if dataset == 'cifar':
        size = 32
    elif dataset == 'clothing1m':
        size = 224
    elif dataset == 'animal10n':
        size = 64
    if cam1.max() == 0 and cam2.max() == 0:
        return torch.tensor(size * np.random.uniform(0.3, 0.8), device=cam1.device)
    elif cam1.max() == 0 or cam2.max() == 0:
        return torch.tensor(size * np.random.uniform(0.3, 0.8),
                            device=cam1.device)

    # compute centroids (vectorized)
    def get_centroid(cam):
        h, w = cam.shape
        # precompute and cache the coordinate grid
        if not hasattr(calc_displacement, 'X_cache') or calc_displacement.X_cache.shape != (h, w):
            x = torch.arange(w, device=cam.device)
            y = torch.arange(h, device=cam.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            calc_displacement.X_cache = X
            calc_displacement.Y_cache = Y

        total = torch.sum(cam)
        cx = torch.sum(calc_displacement.X_cache * cam) / total
        cy = torch.sum(calc_displacement.Y_cache * cam) / total
        return (cx, cy)

    centroid1 = get_centroid(cam1)
    centroid2 = get_centroid(cam2)

    # avoid creating a new tensor
    return torch.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)


def calc_centroid_displacement(cams1, cams2, dataset='clothing1m'):
    # compute centroid displacement for the whole batch
    batch_size = cams1.size(0)
    results = torch.zeros(batch_size, device=cams1.device)
    for i in range(batch_size):
        results[i] = calc_displacement(cams1[i], cams2[i],dataset=dataset)

    return results

