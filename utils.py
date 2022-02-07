import matplotlib.pyplot as plt
import torch
import numpy as np

import time

from torchvision.utils import draw_bounding_boxes, make_grid

EPS = 1e-6

def make_heatmap_unused(mask, size_tensor):
    """
    :param mask: mask size batch_size x num_classes x H x W
    :param size_tensor: size_tensor size batch_size x 2 x H x W, 0 dim for y, 1 dim for x
    :return:
    """
    idxs = (mask == 1).nonzero()
    res_mask = torch.zeros_like(mask)
    
    for b, c, y, x in idxs:
        r = find_radius(size_tensor[b, 0, y, x], size_tensor[b, 1, y, x]) + EPS
                
        x_min, x_max = torch.max(torch.tensor(0.), x - r).floor().long(), \
                       torch.min(x + r, torch.tensor(mask.shape[3])).floor().long()
        y_min, y_max = torch.max(torch.tensor(0.), y - r).floor().long(), \
                       torch.min(y + r, torch.tensor(mask.shape[2])).floor().long()
        
        if x_max == mask.shape[3]:
            x_max -= 1
        if y_max == mask.shape[2]:
            y_max -= 1
        
        x_mean = torch.div(x_min + x_max, 2, rounding_mode='floor')
        y_mean = torch.div(y_min + y_max, 2, rounding_mode='floor')
        
        y_grid = torch.arange(y_min - y_mean, y_max - y_mean + 1, device='cuda').reshape(-1, 1)
        x_grid = torch.arange(x_min - x_mean, x_max - x_mean + 1, device='cuda').reshape(1, -1)

        c_mask = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * r ** 2))
        res_mask[b, c, y_min:y_max + 1, x_min:x_max + 1] = torch.maximum(
            c_mask, res_mask[b, c, y_min:y_max + 1, x_min:x_max + 1]
        )
        res_mask[b, c, y, x] = 1

    return res_mask

def make_heatmap(w, h, c_x, c_y, y_shape, x_shape):
    r = find_radius(w, h) + 1e-6
    
    res_mask = np.zeros((y_shape, x_shape))
    
    x_min, x_max = max(0, int(c_x - r + 0.5)), min(int(c_x + r + 0.5), x_shape)
    y_min, y_max = max(0, int(c_y - r + 0.5)), min(int(c_y + r + 0.5), y_shape)
    
    if x_max == x_shape:
        x_max -= 1
    if y_max == y_shape:
        y_max -= 1

    x_mean, y_mean = (x_min + x_max) // 2, (y_min + y_max) // 2

    y_grid, x_grid = np.ogrid[
        y_min - y_mean:y_max - y_mean + 1, x_min - x_mean:x_max - x_mean + 1
    ]
    c_mask = np.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * r ** 2))
    res_mask[y_min:y_max + 1, x_min:x_max + 1] = c_mask
    res_mask[c_y, c_x] = 1
    
    return res_mask


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
    
    
def make_bin_mask(pred, im_shape, device):
    out = torch.zeros(im_shape, device=device)
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
    inter = ((true_mask * pred_mask) > 0).sum()
    union = ((true_mask + pred_mask) > 0).sum()
    return inter / union


def visualize_batch(model, test_data, count=16, thr=0.3, device='cuda', need_heatmap=False):
    inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                std=[1/0.229, 1/0.224, 1/0.255]
            )
    hm_resize = transforms.Resize(512)

    test_data.return_coords = True
    test_data.bin_mask = False
    
    plt.figure(figsize=(count * 12, count * 8))
    with torch.no_grad():
        for i, data in enumerate(test_data):
            if i >= count:
                break
            img = data[0].to(device)
            pred = model.predict(img.unsqueeze(0), thr=thr).to('cpu')
            
            img_true = (torch.clamp(inv_normalize(img.to('cpu')), 0, 1) * 255).to(torch.uint8)
            img_pred = (torch.clamp(inv_normalize(img.to('cpu')), 0, 1) * 255).to(torch.uint8)
            if need_heatmap:
                img_true = torch.div(img_true, 2, rounding_mode='floor') + \
                    torch.div((hm_resize(data[1]) * 255).to(torch.uint8), 2, rounding_mode='floor'),
                hm, _, _ = model(img.unsqueeze(0))
                img_pred = torch.div(img_pred, 2, rounding_mode='floor') + \
                    torch.div((hm_resize(hm[0]).to('cpu') * 255).to(torch.uint8), 2, rounding_mode='floor'),
                
            img_pred = draw_bounding_boxes(
                img_pred[0],
                pred[:, 2:],
                [str(e) for e in pred[:, 1].to(int).tolist()],
                width=2,
                colors='red'
            )
            img_true = draw_bounding_boxes(
                img_true[0],
                data[2][:, 1:],
                [str(e) for e in data[2][:, 0].to(int).tolist()],
                width=2,
                colors='red'
            )
            plt.subplot(count, 1, i + 1)
            plt.imshow(make_grid([img_true, img_pred], nrow=2).permute(1, 2, 0))
    plt.title('true/pred')
    plt.show()
    
    test_data.return_coords = False
    test_data.bin_mask = True
    
    
def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


class ConfusionMatrix:
    def __init__(self, num_classes: int, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def process_batch(self, det_boxes, det_labels, true_boxes, true_labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 5]), class, x1, y1, x2, y2
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        all_ious = find_jaccard_overlap(true_boxes, det_boxes).cpu().detach().numpy()
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(true_labels):
            gt_class = true_labels[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = det_labels[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        for i, detection in enumerate(det_labels):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = det_labels[i]
                self.matrix[detection_class, self.num_classes] += 1

    def get_matrix(self):
        return self.matrix
    
    def get_precision(self):
        return cm.matrix.diagonal() / cm.matrix.sum(axis=1)
    
    def get_recall(self):
        return cm.matrix.diagonal() / cm.matrix.sum(axis=0)


if __name__ == '__main__':
    from face_dataset.dataset import CenterFaceDataset, center_face_train_test_split

    train_files, test_files, df = center_face_train_test_split(
        helen_path='./data/helen/helen_1',
        fgnet_path='./data/fg_net/images',
        celeba_path='./data/celeba/img_align_celeba'
    )
    ds = CenterFaceDataset(train_files, df)
    visualize_heatmap(ds, 1000)
