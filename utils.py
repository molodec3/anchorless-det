import matplotlib.pyplot as plt
import torch
import numpy as np

EPS = 1e-6


def make_heatmap(mask, size_tensor):
    """
    :param mask: mask size batch_size x num_classes x H x W
    :param size_tensor: size_tensor size batch_size x 2 x H x W, 0 dim for y, 1 dim for x
    :return:
    """
    mask = mask.to('cpu')
    size_tensor = size_tensor.to('cpu')
    idxs = (mask == 1).nonzero(as_tuple=True)
    res_mask = torch.zeros(mask.shape)

    for b, c, y, x in zip(idxs[0], idxs[1], idxs[2], idxs[3]):
        r = find_radius(size_tensor[b, 0, y, x], size_tensor[b, 1, y, x]).item() + EPS

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
                         y_min - y_mean:y_max - y_mean + 1, x_min - x_mean:x_max - x_mean + 1
                         ]
        c_mask = np.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * r ** 2))
        res_mask[b, c, y_min:y_max + 1, x_min:x_max + 1] = np.maximum(
            c_mask, res_mask[b, c, y_min:y_max + 1, x_min:x_max + 1]
        )
        res_mask[b, c, y, x] = 1
    return res_mask.to('cuda')


def find_radius(w, h, min_iou=0.7):
    c_min = min(w, h)
    return (1 - min_iou) * c_min / (1 + min_iou)


def visualize_heatmap(dataset, idx=None):
    if idx is None:
        idx = np.random.randint(0, len(dataset))

    im, mask, size = dataset[idx]
    heatmap = make_heatmap(mask.unsqueeze(0), size.unsqueeze(0))
    plt.subplot(1, 2, 1)
    plt.imshow(im.permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap.squeeze(0).sum(dim=0))
    plt.show()
    
    
def make_bin_mask(pred, im_shape):
    out = torch.zeros(im_shape)
    for b, c, x_min, y_min, x_max, y_max in pred:
        out[int(b), int(c), int(y_min):int(y_max), int(x_min):int(x_max)] = 1
    
    return out


def make_bboxes(pred, im_shape):
    out = torch.zeros(im_shape)
    for b, c, x_min, y_min, x_max, y_max in pred:
        out[int(b), int(c), int(y_min):int(y_max), int(x_min)] = 1
        out[int(b), int(c), int(y_min):int(y_max), int(x_max)] = 1
        out[int(b), int(c), int(y_min), int(x_min):int(x_max)] = 1
        out[int(b), int(c), int(y_max), int(x_min):int(x_max)] = 1
        
    return out


def calculate_iou(true_mask, pred_mask):
    inter = (true_mask * pred_mask > 0).sum()
    union = (true_mask + pred_mask > 0).sum()
    return inter / union


def visualize_prediction(pred):
    pass


if __name__ == '__main__':
    from face_dataset.dataset import CenterFaceDataset, center_face_train_test_split

    train_files, test_files, df = center_face_train_test_split(
        helen_path='./data/helen/helen_1',
        fgnet_path='./data/fg_net/images',
        celeba_path='./data/celeba/img_align_celeba'
    )
    ds = CenterFaceDataset(train_files, df)
    visualize_heatmap(ds, 1000)
