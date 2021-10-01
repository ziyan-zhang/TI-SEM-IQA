"""-----------------------------------------------------
创建时间 :  2020/5/6  15:47
说明    :
todo   :
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
import os.path as osp
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import cv2
from tqdm import tqdm


def make_dataset(directory, opt):
    # opt: 'train' or 'val'
    # glob.glob中, *代表任意字符, 用于匹配
    # 自定义的数据集不同的文件夹里面的文件不一样
    masked = glob.glob(osp.join(directory, opt + '_kthinm/*.png'))
    # masked = glob.glob(osp.join(directory, opt + '_thinin/*.png'))

    masked = sorted(masked)

    real_center = glob.glob(osp.join(directory, opt + '_center/*.png'))
    real_center = sorted(real_center)

    return list(zip(masked, real_center))

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # return img.convert('RGB')
            return img.convert('L')


def default_loader(path):
    return pil_loader(path)

class ImageFolder(data.Dataset):
    def __init__(self, opt, root, transform=None):

        self.root = root
        self.imgs = make_dataset(root, opt)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        masked_path, real_center_path = self.imgs[index]
        masked = Image.open(masked_path)
        real_center = Image.open(real_center_path)

        if self.transform is not None:
            masked, real_center = self.transform([masked, real_center])
        masked, real_center = masked.float(), real_center.float()
        return masked, real_center

    def __len__(self):
        return len(self.imgs)

