import os

from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def preprocess_helen(path_files, path_annotations, path_out,
                     padding_eye=0.1, padding_mouth=0.1, show_test=False):
    if not isinstance(padding_eye, tuple):
        padding_eye = (padding_eye, padding_eye)
    if not isinstance(padding_mouth, tuple):
        padding_mouth = (padding_mouth, padding_mouth)

    annotations = os.listdir(path_annotations)

    for k, a in tqdm(enumerate(annotations)):
        with open(f'{path_annotations}/{a}', 'r') as f:
            c_file = ''
            x_min_eye, x_max_eye = float('inf'), -float('inf')
            y_min_eye, y_max_eye = float('inf'), -float('inf')

            x_min_mouth, x_max_mouth = float('inf'), -float('inf')
            y_min_mouth, y_max_mouth = float('inf'), -float('inf')

            for i, line in enumerate(f):
                if i == 0:
                    c_file = line[:-1]
                    continue
                elif 115 <= i <= 154:
                    c_x, c_y = map(float, line.split(' , '))
                    if c_x > x_max_eye:
                        x_max_eye = c_x
                    elif c_x < x_min_eye:
                        x_min_eye = c_x

                    if c_y > y_max_eye:
                        y_max_eye = c_y
                    elif c_y < y_min_eye:
                        y_min_eye = c_y

                elif 59 <= i <= 114:
                    c_x, c_y = map(float, line.split(' , '))
                    if c_x > x_max_mouth:
                        x_max_mouth = c_x
                    elif c_x < x_min_mouth:
                        x_min_mouth = c_x

                    if c_y > y_max_mouth:
                        y_max_mouth = c_y
                    elif c_y < y_min_mouth:
                        y_min_mouth = c_y

        im = Image.open(f'{path_files}/{c_file}.jpg')
        x_im, y_im = im.size

        x_pad, y_pad = padding_eye
        x_dist_eye = x_max_eye - x_min_eye
        y_dist_eye = y_max_eye - y_min_eye
        x_max_eye = min(x_max_eye + x_pad * x_dist_eye, x_im)
        x_min_eye = max(x_min_eye - x_pad * x_dist_eye, 0)
        y_max_eye = min(y_max_eye + y_pad * y_dist_eye, y_im)
        y_min_eye = max(y_min_eye - y_pad * y_dist_eye, 0)

        x_pad, y_pad = padding_mouth
        x_dist_mouth = x_max_mouth - x_min_mouth
        y_dist_mouth = y_max_mouth - y_min_mouth
        x_max_mouth = min(x_max_mouth + x_pad * x_dist_mouth, x_im)
        x_min_mouth = max(x_min_mouth - x_pad * x_dist_mouth, 0)
        y_max_mouth = min(y_max_mouth + y_pad * y_dist_mouth, y_im)
        y_min_mouth = max(y_min_mouth - y_pad * y_dist_mouth, 0)

        with open(path_out, 'w' if k == 0 else 'a') as f:
            f.write(f'{c_file},eyes,{x_min_eye},{y_min_eye},{x_max_eye},{y_max_eye}\n')
            f.write(f'{c_file},mouth,{x_min_mouth},{y_min_mouth},{x_max_mouth},{y_max_mouth}\n')

        if show_test and k % 180 == 0:
            plt.imshow(im)
            plt.plot([x_min_eye, x_min_eye, x_max_eye, x_max_eye, x_min_eye],
                     [y_min_eye, y_max_eye, y_max_eye, y_min_eye, y_min_eye])
            plt.plot([x_min_mouth, x_min_mouth, x_max_mouth, x_max_mouth, x_min_mouth],
                     [y_min_mouth, y_max_mouth, y_max_mouth, y_min_mouth, y_min_mouth])
            plt.show()


def preprocess_fgnet(path_files, path_annotations, path_out,
                     padding_eye=0.1, padding_mouth=0.1, show_test=False):
    if not isinstance(padding_eye, tuple):
        padding_eye = (padding_eye, padding_eye)
    if not isinstance(padding_mouth, tuple):
        padding_mouth = (padding_mouth, padding_mouth)

    annotations = os.listdir(path_annotations)

    for k, a in tqdm(enumerate(annotations)):
        with open(f'{path_annotations}/{a}', 'r') as f:
            c_file = a.split('.')[0]

            x_min_eye, x_max_eye = float('inf'), -float('inf')
            y_min_eye, y_max_eye = float('inf'), -float('inf')

            x_min_mouth, x_max_mouth = float('inf'), -float('inf')
            y_min_mouth, y_max_mouth = float('inf'), -float('inf')

            for i, line in enumerate(f):
                if line == '}\n':
                    continue
                if 30 <= i < 40:
                    c_x, c_y = map(float, line.split(' '))
                    if c_x > x_max_eye:
                        x_max_eye = c_x
                    elif c_x < x_min_eye:
                        x_min_eye = c_x

                    if c_y > y_max_eye:
                        y_max_eye = c_y
                    elif c_y < y_min_eye:
                        y_min_eye = c_y

                elif 50 < i < 70:
                    c_x, c_y = map(float, line.split(' '))
                    if c_x > x_max_mouth:
                        x_max_mouth = c_x
                    elif c_x < x_min_mouth:
                        x_min_mouth = c_x

                    if c_y > y_max_mouth:
                        y_max_mouth = c_y
                    elif c_y < y_min_mouth:
                        y_min_mouth = c_y

        im = Image.open(f'{path_files}/{c_file[:6].upper() + c_file[6:]}.JPG')
        x_im, y_im = im.size

        x_pad, y_pad = padding_eye
        x_dist_eye = x_max_eye - x_min_eye
        y_dist_eye = y_max_eye - y_min_eye
        x_max_eye = min(x_max_eye + x_pad * x_dist_eye, x_im)
        x_min_eye = max(x_min_eye - 2 * x_pad * x_dist_eye, 0)
        y_max_eye = min(y_max_eye + y_pad * y_dist_eye, y_im)
        y_min_eye = max(y_min_eye - y_pad * y_dist_eye, 0)

        x_pad, y_pad = padding_mouth
        x_dist_mouth = x_max_mouth - x_min_mouth
        y_dist_mouth = y_max_mouth - y_min_mouth
        x_max_mouth = min(x_max_mouth + x_pad * x_dist_mouth, x_im)
        x_min_mouth = max(x_min_mouth - 2 * x_pad * x_dist_mouth, 0)
        y_max_mouth = min(y_max_mouth + 2 * y_pad * y_dist_mouth, y_im)
        y_min_mouth = max(y_min_mouth - y_pad * y_dist_mouth, 0)

        with open(path_out, 'w' if k == 0 else 'a') as f:
            f.write(f'{c_file},eyes,{x_min_eye},{y_min_eye},{x_max_eye},{y_max_eye}\n')
            f.write(f'{c_file},mouth,{x_min_mouth},{y_min_mouth},{x_max_mouth},{y_max_mouth}\n')

        if show_test and k % 95 == 0:
            plt.imshow(im)
            plt.plot([x_min_eye, x_min_eye, x_max_eye, x_max_eye, x_min_eye],
                     [y_min_eye, y_max_eye, y_max_eye, y_min_eye, y_min_eye])
            plt.plot([x_min_mouth, x_min_mouth, x_max_mouth, x_max_mouth, x_min_mouth],
                     [y_min_mouth, y_max_mouth, y_max_mouth, y_min_mouth, y_min_mouth])
            plt.show()


def preprocess_celeba(path_files, path_annotations, path_out,
                      padding_eye=0.1, padding_mouth=0.1, show_test=False):
    if not isinstance(padding_eye, tuple):
        padding_eye = (padding_eye, padding_eye)
    if not isinstance(padding_mouth, tuple):
        padding_mouth = (padding_mouth, padding_mouth)

    with open(path_annotations, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            pic_arr = line.split()
            c_file = pic_arr[0]
            coord = list(map(int, pic_arr[1:]))
            coord_x, coord_y = coord[::2], coord[1::2]
            x_eye, y_eye = coord_x[36:48], coord_y[36:48]
            x_mouth, y_mouth = coord_x[48:], coord_y[48:]

            x_max_eye, x_min_eye = max(x_eye), min(x_eye)
            y_max_eye, y_min_eye = max(y_eye), min(y_eye)
            x_max_mouth, x_min_mouth = max(x_mouth), min(x_mouth)
            y_max_mouth, y_min_mouth = max(y_mouth), min(y_mouth)

            im = Image.open(f'{path_files}/{c_file}')
            x_im, y_im = im.size

            x_pad, y_pad = padding_eye
            x_dist_eye = x_max_eye - x_min_eye
            y_dist_eye = y_max_eye - y_min_eye
            x_max_eye = min(x_max_eye + x_pad * x_dist_eye, x_im)
            x_min_eye = max(x_min_eye - x_pad * x_dist_eye, 0)
            y_max_eye = min(y_max_eye + y_pad * y_dist_eye, y_im)
            y_min_eye = max(y_min_eye - y_pad * y_dist_eye, 0)

            x_pad, y_pad = padding_mouth
            x_dist_mouth = x_max_mouth - x_min_mouth
            y_dist_mouth = y_max_mouth - y_min_mouth
            x_max_mouth = min(x_max_mouth + x_pad * x_dist_mouth, x_im)
            x_min_mouth = max(x_min_mouth - x_pad * x_dist_mouth, 0)
            y_max_mouth = min(y_max_mouth + y_pad * y_dist_mouth, y_im)
            y_min_mouth = max(y_min_mouth - y_pad * y_dist_mouth, 0)

            with open(path_out, 'w' if i == 0 else 'a') as f:
                if i == 0:
                    f.write('name,class,x_min,y_min,x_max,y_max')
                f.write(f'{c_file.split(".")[0]},eyes,{x_min_eye},{y_min_eye},{x_max_eye},{y_max_eye}\n')
                f.write(f'{c_file.split(".")[0]},mouth,{x_min_mouth},{y_min_mouth},{x_max_mouth},{y_max_mouth}\n')

            if show_test and i % 1203 == 0:
                plt.imshow(im)
                plt.plot([x_min_eye, x_min_eye, x_max_eye, x_max_eye, x_min_eye],
                         [y_min_eye, y_max_eye, y_max_eye, y_min_eye, y_min_eye])
                plt.plot([x_min_mouth, x_min_mouth, x_max_mouth, x_max_mouth, x_min_mouth],
                         [y_min_mouth, y_max_mouth, y_max_mouth, y_min_mouth, y_min_mouth])
                plt.show()


if __name__ == '__main__':
    preprocess_helen('helen/helen_1',
                     'helen/annotation',
                     'helen/pascal_annotation.csv',
                     padding_eye=(0.1, 0.3), padding_mouth=0.1, show_test=True)

    preprocess_fgnet('fg_net/images',
                     'fg_net/points',
                     'fg_net/pascal_annotation.csv',
                     padding_eye=(0.1, 0.3), padding_mouth=0.1, show_test=True)

    preprocess_celeba('celeba/img_align_celeba',
                      'celeba/landmark_align.txt',
                      'celeba/pascal_annotation.csv',
                      padding_eye=(0.1, 0.4), padding_mouth=0.1, show_test=True)
