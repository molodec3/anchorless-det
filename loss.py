import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import make_heatmap

EPS = 1e-6


def focal_loss(pred, truth, mask, alpha, beta):
    """
    :param pred: model prediction heatmap batch_size x classes x H x W
    :param truth: ground truth heatmap batch_size x classes x H x W
    :param mask: truth mask batch_size x classes x H x W
    :param alpha: hyper parameter alpha
    :param beta: hyper parameter beta
    :return: focal loss
    """

    pos_loss = (torch.pow(1 - pred, alpha) * torch.log(pred) * mask).sum()
    neg_loss = (torch.pow(1 - truth, beta) * torch.pow(pred, alpha) * torch.log(1 - pred) * (1 - mask)).sum()

    non_zero_count = mask.sum()
    if non_zero_count == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / non_zero_count

    return loss


def offset_loss(pred, truth, mask):
    """
    :param pred: model prediction offset batch_size x 2 x H x W, 0 dim for y, 1 dim for x
    :param truth: ground truth offset batch_size x 2 x H x W, 0 dim for y, 1 dim for x
    :param mask: truth mask
    :return: offset_loss
    """

    loss = torch.abs((pred - truth) * mask).sum() / (mask.sum() + EPS)

    return loss


def size_loss(pred, truth, mask):
    """
    :param pred: model prediction size batch_size x 2 x H x W, 0 dim for y, 1 dim for x
    :param truth: ground truth size batch_size x 2 x H x W, 0 dim for y, 1 dim for x
    :param mask: truth mask
    :return: size_loss
    """
    loss = torch.abs((pred - truth) * mask).sum() / (mask.sum() + EPS)

    return loss


def center_net_loss(
        pred_heatmap, pred_offset, pred_size,
        heatmap, mask, size_tensor, offset_tensor,
        alpha=2, beta=4,
        lambd_offset=1, lambd_size=0.1,
        downsampling_ratio=4
):
    """
    :param pred_heatmap: predicted heatmap batch_size x num_classes x H x W
    :param pred_offset: predicted offset batch_size x num_classes x H x W
    :param pred_size: predicted size batch_size x num_classes x H x W
    :param mask: real mask batch_size x num_classes x (H * R) x (W * R)
    :param size_tensor: real box size batch_size x 2 x (H * R) x (W * R), 0 dim for y, 1 dim for x
    :param alpha:
    :param beta:
    :param lambd_offset:
    :param lambd_size:
    :param downsampling_ratio:
    :return:
    """
    if isinstance(alpha, int):
        alpha = torch.tensor(alpha, device='cuda')
    if isinstance(beta, int):
        beta = torch.tensor(beta, device='cuda')
    focal = focal_loss(pred_heatmap, heatmap, mask, alpha, beta)

    # merged mask without respect to classes for size and offset
    merged_mask = mask.sum(dim=1, keepdim=True)

    offset = offset_loss(pred_offset, offset_tensor, merged_mask)

    size = size_loss(pred_size, size_tensor, merged_mask)
    
    return focal + lambd_offset * offset + lambd_size * size
