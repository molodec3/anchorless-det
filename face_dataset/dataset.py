import os

import albumentations as A
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections.abc import Iterable
from matplotlib.patches import Rectangle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2


class CenterFaceDataset(Dataset):
    def __init__(self, dataset_limit=5000):
        self.transform = A.Compose(
            [
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
        )

        self.classes = {'eyes', 'mouth'}
        self.n_classes = 2
        self.idxs_classes = {0: 'eyes', 1: 'mouth'}
        self.classes_idxs = {'eyes': 0, 'mouth': 1}

        helen_path = '/home/andrey/Downloads/face_datasets/helen/helen_1'
        fgnet_path = '/home/andrey/Downloads/face_datasets/fg_net/images'
        celeba_path = '/home/andrey/Downloads/face_datasets/celeba/img_align_celeba'

        helen_files = os.listdir(helen_path)
        self.files = [f'{helen_path}/{f}' for f in helen_files]
        fgnet_files = os.listdir(fgnet_path)
        self.files.extend([f'{fgnet_path}/{f}' for f in fgnet_files])
        celeba_size = max(dataset_limit - len(self.files), 0)
        celeba_files = np.random.choice(os.listdir(celeba_path), celeba_size, replace=False).tolist()
        self.files.extend([f'{celeba_files}/{f}' for f in celeba_files])

        helen_bbox = pd.read_csv(f'{helen_path}/../pascal_annotation.csv', dtype={'name': str})
        helen_bbox = helen_bbox.set_index('name')
        fgnet_bbox = pd.read_csv(f'{fgnet_path}/../pascal_annotation.csv', dtype={'name': str})
        fgnet_bbox = fgnet_bbox.set_index('name')
        celeba_bbox = pd.read_csv(f'{celeba_path}/../pascal_annotation.csv', dtype={'name': str})
        celeba_bbox = celeba_bbox.set_index('name')

        self.data_df = pd.concat([helen_bbox, fgnet_bbox, celeba_bbox], axis=0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file.split('/')[-1].split('.')[0]
        df = self.data_df.loc[name]
        print(df.iloc[:, -4:].to_numpy())
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(img)
        # plt.scatter(df.loc[:, 'x_min'], df.loc[:, 'y_min'])
        # plt.scatter(df.loc[:, 'x_min'], df.loc[:, 'y_max'])
        # plt.scatter(df.loc[:, 'x_max'], df.loc[:, 'y_min'])
        # plt.scatter(df.loc[:, 'x_max'], df.loc[:, 'y_max'])
        # plt.show()
        transformed = \
            self.transform(
                image=img, bboxes=df.iloc[:, -4:].to_numpy(), class_labels=df.iloc[:, 0].to_numpy()
            )
        print(type(transformed['image']))
        # plt.imshow(transformed['image'])
        # plt.scatter(transformed['bboxes'][0][0], transformed['bboxes'][0][1])
        # plt.scatter(transformed['bboxes'][0][0], transformed['bboxes'][0][3])
        # plt.scatter(transformed['bboxes'][0][2], transformed['bboxes'][0][1])
        # plt.scatter(transformed['bboxes'][0][2], transformed['bboxes'][0][3])
        # plt.scatter(transformed['bboxes'][1][0], transformed['bboxes'][1][1])
        # plt.scatter(transformed['bboxes'][1][0], transformed['bboxes'][1][3])
        # plt.scatter(transformed['bboxes'][1][2], transformed['bboxes'][1][1])
        # plt.scatter(transformed['bboxes'][1][2], transformed['bboxes'][1][3])
        # plt.show()

        img = transformed['image']
        mask = torch.zeros(self.n_classes, img.shape[1], img.shape[2])
        size = torch.zeros(2, img.shape[1], img.shape[2])
        for (x_min, y_min, x_max, y_max), c in zip(transformed['bboxes'], transformed['class_labels']):
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            mask[self.classes_idxs[c], center_x, center_y] = 1
            # check if x is 0 dim and y is 1 dim
            size[0, center_x, center_y] = x_max - x_min
            size[1, center_x, center_y] = y_max - y_min

        return img, mask, size
        # return only image, non-strided mask, size tensor


if __name__ == '__main__':
    ds = CenterFaceDataset()
    ds[100]
