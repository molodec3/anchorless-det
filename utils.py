import torch
import numpy as np

EPS = 1e-6


def make_heatmap(mask, size_tensor):
    """
    :param mask: mask size batch_size x num_classes x H x W
    :param size_tensor: size_tensor size batch_size x 2 x H x W, 0 dim for y, 1 dim for x
    :return:
    """
    idxs = (mask == 1).nonzero(as_tuple=True)
    res_mask = torch.zeros(mask.shape)

    for i in range(mask.shape[1]):
        for b, y, x in zip(idxs[0], idxs[2], idxs[3]):
            r = find_radius(size_tensor[i, 0, y, x], size_tensor[i, 1, y, x]).item() + EPS

            x_min, x_max = torch.max(torch.tensor(0.), x - r).floor().long().item(), \
                           torch.min(x + r, torch.tensor(mask.shape[3])).floor().long().item()
            y_min, y_max = torch.max(torch.tensor(0.), y - r).floor().long().item(), \
                           torch.min(y + r, torch.tensor(mask.shape[2])).floor().long().item()

            if x_max == mask.shape[3]:
                x_max -= 1
            if y_max == mask.shape[2]:
                y_max -= 1

            x_mean, y_mean = (x_min + x_max) // 2, (y_min + y_max) // 2

            y_grid, x_grid = np.ogrid[
                             y_min-y_mean:y_max-y_mean+1, x_min-x_mean:x_max-x_mean+1
                             ]
            c_mask = np.exp(-(x_grid**2 + y_grid**2) / (2 * r**2))
            res_mask[b, i, y_min:y_max+1, x_min:x_max+1] = np.maximum(
                c_mask, res_mask[b, i, y_min:y_max+1, x_min:x_max+1]
            )
            res_mask[b, i, y, x] = 1
    return res_mask


def find_radius(w, h, min_iou=0.7):
    c_min = min(w, h)
    return (1 - min_iou) * c_min / (1 + min_iou)
