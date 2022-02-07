import albumentations as A
import torch
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from anchorless_det.utils import make_heatmap


class CocoDataset(Dataset):
    def __init__(
            self, files, folder, transforms=None, bin_mask=False, need_classes=False, downsampling_ratio=4
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.downsampling_ratio = downsampling_ratio
        self.folder = folder
        self.bin_mask = bin_mask  # for val
        self.return_coords = False
        
        if transforms is None:
            self.transform = A.Compose(
                [
                    A.Resize(512, 512),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']),
            )
        else:
            self.transform = transforms

        self.need_classes = need_classes
        
        with open(files, 'r') as f:
            files_parsed = json.loads(f.read())
        self.images_w_bboxes = list(files_parsed['img_bbox'].items())
        self.images_w_bboxids = files_parsed['img_bboxid']
        self.classes = files_parsed['bboxid_catid']
        self.img_names = files_parsed['img_name']
        
        if self.need_classes:
            self.n_classes = len(files_parsed['catid_catname'])
            self.classes_num = {catid: i for i, catid in enumerate(files_parsed['catid_catname'].keys())}
        else:
            self.n_classes = 1

    def __len__(self):
        return len(self.images_w_bboxes)

    def __getitem__(self, idx):
        file, bboxes = self.images_w_bboxes[idx]
        img = cv2.imread(f'{self.folder}/{self.img_names[file]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.need_classes:
            classes = []
            ids = self.images_w_bboxids[file]
            for bbox_id in ids:
                classes.append(self.classes_num[str(self.classes[str(bbox_id)])])
        else:
            classes = [0 for _ in range(len(self.images_w_bboxids[file]))]
        
        transformed = \
            self.transform(
                image=img, bboxes=bboxes, class_labels=classes
            )

        img = transformed['image']
        downsampled_h, downsampled_w = img.shape[1] // self.downsampling_ratio, img.shape[2] // self.downsampling_ratio
        mask = torch.zeros(
            self.n_classes, downsampled_h, downsampled_w
        )
        heatmap = torch.zeros(
            self.n_classes, downsampled_h, downsampled_w
        )
        if self.bin_mask:
            bin_mask = torch.zeros(self.n_classes, img.shape[1], img.shape[2])
        if self.return_coords:
            coords = []
        size = torch.zeros(
            2, downsampled_h, downsampled_w
        )
        offset = torch.zeros(
            2, downsampled_h, downsampled_w
        )
        for (x_min, y_min, w, h), c in zip(transformed['bboxes'], transformed['class_labels']):
            if self.return_coords:
                coords.append(torch.tensor([[c, x_min, y_min, x_min + w, y_min + h]]))
            
            center_x, center_y = int(x_min + w / 2), int(y_min + h / 2)
            mask[
                c, 
                center_y // self.downsampling_ratio, 
                center_x // self.downsampling_ratio
            ] = 1
            if self.bin_mask:
                bin_mask[c, int(y_min):int(y_min + h), int(x_min):int(x_min + w)] = 1
            # y is 0 dim and x is 1 dim
            size[
                0, 
                center_y // self.downsampling_ratio, 
                center_x // self.downsampling_ratio
            ] = h // self.downsampling_ratio
            size[
                1, 
                center_y // self.downsampling_ratio, 
                center_x // self.downsampling_ratio
            ] = w // self.downsampling_ratio
            
            offset[
                0, 
                center_y // self.downsampling_ratio,
                center_x // self.downsampling_ratio
            ] = h / self.downsampling_ratio - h // self.downsampling_ratio
            offset[
                1,
                center_y // self.downsampling_ratio,
                center_x // self.downsampling_ratio
            ] = w / self.downsampling_ratio - w // self.downsampling_ratio
            
            c_hm = make_heatmap(
                w // self.downsampling_ratio, 
                h // self.downsampling_ratio, 
                center_x // self.downsampling_ratio, 
                center_y // self.downsampling_ratio, 
                downsampled_h, downsampled_w
            )
            heatmap[c] = np.maximum(
                c_hm,
                heatmap[c]
            )

        # if bin_mask required than return bin_mask of bboxes instead of mask with centers
        if self.bin_mask:
            return img, heatmap, mask, size, offset, bin_mask
        
        if self.return_coords:
            return img, heatmap, torch.cat(coords, dim=0)

        # return only image, non-strided mask, size tensor
        return img, heatmap, mask, size, offset
