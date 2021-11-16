import os
from copy import deepcopy

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
from sklearn.model_selection import train_test_split


class CenterFaceDataset(Dataset):
    def __init__(
            self, files, df, dataset_limit=5000, bin_mask=False
    ):
        self.plot_im = False  # for debug mostly
        
        self.bin_mask = bin_mask  # for val

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

        self.files = files

        self.data_df = df

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file.split('/')[-1].split('.')[0].lower()
        df = self.data_df.loc[name]
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = \
            self.transform(
                image=img, bboxes=df.iloc[:, -4:].to_numpy(), class_labels=df.iloc[:, 0].to_numpy()
            )

        if self.plot_im:
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.scatter(df.loc[:, 'x_min'], df.loc[:, 'y_min'])
            plt.scatter(df.loc[:, 'x_min'], df.loc[:, 'y_max'])
            plt.scatter(df.loc[:, 'x_max'], df.loc[:, 'y_min'])
            plt.scatter(df.loc[:, 'x_max'], df.loc[:, 'y_max'])

            plt.subplot(1, 2, 2)
            plt.imshow(transformed['image'].permute(1, 2, 0))
            plt.scatter(transformed['bboxes'][0][0], transformed['bboxes'][0][1])
            plt.scatter(transformed['bboxes'][0][0], transformed['bboxes'][0][3])
            plt.scatter(transformed['bboxes'][0][2], transformed['bboxes'][0][1])
            plt.scatter(transformed['bboxes'][0][2], transformed['bboxes'][0][3])
            plt.scatter(transformed['bboxes'][1][0], transformed['bboxes'][1][1])
            plt.scatter(transformed['bboxes'][1][0], transformed['bboxes'][1][3])
            plt.scatter(transformed['bboxes'][1][2], transformed['bboxes'][1][1])
            plt.scatter(transformed['bboxes'][1][2], transformed['bboxes'][1][3])

            plt.show()

        img = transformed['image']
        mask = torch.zeros(self.n_classes, img.shape[1], img.shape[2])
        if self.bin_mask:
            bin_mask = torch.zeros(self.n_classes, img.shape[1], img.shape[2])
        size = torch.zeros(2, img.shape[1], img.shape[2])
        for (x_min, y_min, x_max, y_max), c in zip(transformed['bboxes'], transformed['class_labels']):
            center_x, center_y = int((x_min + x_max) // 2), int((y_min + y_max) / 2)
            mask[self.classes_idxs[c], center_y, center_x] = 1
            if self.bin_mask:
                bin_mask[self.classes_idxs[c], int(y_min):int(y_max), int(x_min):int(x_max)] = 1
            # y is 0 dim and x is 1 dim
            size[0, center_y, center_x] = y_max - y_min
            size[1, center_y, center_x] = x_max - x_min

        # if bin_mask required than return bin_mask of bboxes instead of mask with centers
        if self.bin_mask:
            return img, mask, size, bin_mask

        # return only image, non-strided mask, size tensor
        return img, mask, size


def center_face_train_test_split(
        helen_path='../data/helen/helen_1',
        fgnet_path='../data/fg_net/images',
        celeba_path='../data/celeba/img_align_celeba',
        dataset_limit=5000, test_size=0.2, random_state=None
):
    helen_files = os.listdir(helen_path)
    files = [f'{helen_path}/{f}' for f in helen_files]
    fgnet_files = os.listdir(fgnet_path)
    files.extend([f'{fgnet_path}/{f}' for f in fgnet_files])
    celeba_size = max(dataset_limit - len(files), 0)
    celeba_files = np.random.choice(os.listdir(celeba_path), celeba_size, replace=False).tolist()
    files.extend([f'{celeba_path}/{f}' for f in celeba_files])

    helen_bbox = pd.read_csv(f'{helen_path}/../pascal_annotation.csv', dtype={'name': str})
    helen_bbox = helen_bbox.set_index('name')
    fgnet_bbox = pd.read_csv(f'{fgnet_path}/../pascal_annotation.csv', dtype={'name': str})
    fgnet_bbox = fgnet_bbox.set_index('name')
    celeba_bbox = pd.read_csv(f'{celeba_path}/../pascal_annotation.csv', dtype={'name': str})
    celeba_bbox = celeba_bbox.set_index('name')

    df = pd.concat([helen_bbox, fgnet_bbox, celeba_bbox], axis=0)

    train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)
    return train_files, test_files, df


def main_test():
    a, b, df = center_face_train_test_split()
    a = CenterFaceDataset(a, df)
    b = CenterFaceDataset(b, df)
    a.plot_im = True
    b.plot_im = True
    print(len(a), len(b))
    a[100]
    b[200]

if __name__ == '__main__':
    main_test()
