import os

import albumentations as A
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections.abc import Iterable
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CenterFaceDataset(Dataset):
    def __init__(self, dataset_limit=5000):
        self.classes = {'eyes', 'mouth'}
        self.map_classes = {0: 'eyes', 1: 'mouth'}
        helen_path = '/home/andrey/Downloads/face_datasets/helen/helen_1'
        fgnet_path = '/home/andrey/Downloads/face_datasets/fg_net/images'
        celeba_path = '/home/andrey/Downloads/face_datasets/celeba/img_align_celeba'

        helen_files = os.listdir(helen_path)
        self.files = helen_files[:]
        fgnet_files = os.listdir(fgnet_path)
        self.files.extend(fgnet_files)
        celeba_size = max(dataset_limit - len(self.files), 0)
        celeba_files = np.random.choice(os.listdir(celeba_path), celeba_size, replace=False).tolist()
        self.files.extend(celeba_files)

        helen_bbox = pd.read_csv(f'{helen_path}/../pascal_annotation.csv', dtype={'name': str})
        helen_bbox = helen_bbox.set_index('name')
        fgnet_bbox = pd.read_csv(f'{fgnet_path}/../pascal_annotation.csv', dtype={'name': str})
        fgnet_bbox = fgnet_bbox.set_index('name')
        celeba_bbox = pd.read_csv(f'{celeba_path}/../pascal_annotation.csv', dtype={'name': str})
        celeba_bbox = celeba_bbox.set_index('name')

        self.classes = helen_bbox.loc[list(map(lambda x: x.split('.')[0], helen_files)), 'class'].to_list()
        self.classes.extend(fgnet_bbox.loc[list(map(lambda x: x.lower().split('.')[0], fgnet_files)), 'class'].to_list())
        self.classes.extend(celeba_bbox.loc[list(map(lambda x: x.split('.')[0], celeba_files)), 'class'].to_list())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    ds = CenterFaceDataset()
