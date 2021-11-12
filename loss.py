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

    pos_labels = mask == 1
    neg_labels = mask != 1

    loss = torch.pow(1 - pred, alpha) * torch.log(pred) * pos_labels
    loss += torch.pow(1 - truth, beta) * torch.pow(pred, alpha) * torch.log(1 - pred) * neg_labels

    loss = -loss.sum()
    if pos_labels.sum() != 0:
        loss /= pos_labels.sum()

    return loss


def offset_loss(pred, truth, mask):
    """
    :param pred: model prediction offset batch_size x 2 x H x W, 0 dim for y, 1 dim for x
    :param truth: ground truth offset batch_size x 2 x H x W, 0 dim for y, 1 dim for x
    :param mask: truth mask
    :return: offset_loss
    """

    pos_labels = mask == 1
    loss = torch.abs(pred[pos_labels] - truth[pos_labels]).sum() / (pos_labels.sum() + EPS)

    return loss


def size_loss(pred, truth, mask):
    """
    :param pred: model prediction size batch_size x 2 x H x W, 0 dim for y, 1 dim for x
    :param truth: ground truth size batch_size x 2 x H x W, 0 dim for y, 1 dim for x
    :param mask: truth mask
    :return: size_loss
    """
    pos_labels = mask == 1
    loss = torch.abs(pred[pos_labels] - truth[pos_labels]).sum() / (pos_labels.sum() + EPS)

    return loss


def center_net_loss(
        pred_heatmap, pred_offset, pred_size,
        mask, size_tensor,
        device='cuda' if torch.cuda.is_available() else 'cpu',
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

    idxs = (mask == 1).nonzero(as_tuple=True)
    # get downsampled indexes
    idxs_strided = (
        idxs[0], idxs[1],
        (idxs[2].double() / downsampling_ratio).floor().long(),
        (idxs[3].double() / downsampling_ratio).floor().long()
    )
    mask_strided = torch.zeros([
        mask.shape[0], mask.shape[1],
        mask.shape[2] // downsampling_ratio,
        mask.shape[3] // downsampling_ratio
    ]).to(device)
    mask_strided[idxs_strided] = 1

    real_offsets = torch.zeros([
        mask.shape[0], 2,
        mask.shape[2] // downsampling_ratio,
        mask.shape[3] // downsampling_ratio
    ]).to(device)
    real_offsets[:, 0, idxs_strided[2], idxs_strided[3]] = \
        idxs[2].float() / downsampling_ratio - idxs_strided[2]
    real_offsets[:, 1, idxs_strided[2], idxs_strided[3]] = \
        idxs[3].float() / downsampling_ratio - idxs_strided[3]
    
    downsampled_size = torch.zeros([
        mask.shape[0], 2,
        mask.shape[2] // downsampling_ratio,
        mask.shape[3] // downsampling_ratio
    ]).to(device)
    downsampled_size[idxs_strided] = size_tensor[idxs]

    focal = focal_loss(pred_heatmap, make_heatmap(mask_strided, size_tensor), mask_strided, alpha, beta)

    # merged mask without respect to classes for size and offset
    merged_mask = (mask_strided.sum(dim=1)[:, None, :, :] > 0).float().repeat(1, 2, 1, 1)

    offset = offset_loss(pred_offset, real_offsets, merged_mask)

    size = size_loss(pred_size, downsampled_size, merged_mask)

    return focal + lambd_offset * offset + lambd_size * size
