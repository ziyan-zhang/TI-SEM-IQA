"""-----------------------------------------------------
创建时间 :  2020/05/25  21:02
todo    :
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from utils import transforms as custom_transforms
from PIL import Image
from progressbar import *

def get_transforms_train():
    transforms_list = [
        # custom_transforms.RandomHorizontalFlip(),
        custom_transforms.toRGB('RGB'),
        custom_transforms.toTensor(),  # 三个通道的mean, std。顺序不知
    ]
    transformscp = custom_transforms.Compose(transforms_list)
    return transformscp

def get_transforms_val():
    transforms_list = [
        # custom_transforms.RandomHorizontalFlip(),
        custom_transforms.toRGB('RGB'),
        custom_transforms.toTensor(),  # 三个通道的mean, std。顺序不知
    ]
    transformscp = custom_transforms.Compose(transforms_list)
    return transformscp


class MyDataset(Dataset):
    def __init__(self, img_folder, labelFpath, transform):
        super(MyDataset, self).__init__()
        self.img_names = [os.path.join(img_folder, fileName) for fileName in os.listdir(img_folder)]

        label_pd = pd.read_csv(labelFpath, header=None)
        label_array = np.array(label_pd)
        self.label_array = label_array.squeeze()
        self.transform = transform

        self.img_reads = []
        progress = ProgressBar()
        for i in progress(range(len(self.img_names))):
            img_name_i = self.img_names[i]
            img_read_i = Image.open(img_name_i)
            # 三通道的话要加一句
            # img_read_i = img_read_i.convert('RGB')
            if self.transform is not None:
                [img_read_i] = self.transform([img_read_i])
            img_read_i = img_read_i.float()
            self.img_reads.append(img_read_i)

        assert len(self.img_reads) == len(self.label_array)
        print('数据集大小：', len(self.img_reads))

    def __getitem__(self, index):
        return self.img_reads[index], self.label_array[index]

    def __len__(self):
        return len(self.img_reads)

_trainImageFolder = 'E:\\July2_cut_Overlap\\train_img'
_trainLabelPath = 'E:\\July2_cut_Overlap\\mos_train7_16.csv'
_valImageFolder = 'E:\\July2_cut_Overlap\\val_img'
_valLabelPath = 'E:\\July2_cut_Overlap\\mos_val3_16.csv'

_transform_train = get_transforms_train()  # 使用(这里还是自定义的)transform gpu利用率比预先除好tensor好效率更高. 可能是因为自带的torch totensor直接用比先弄好优化得好?
_transform_val = get_transforms_val()  # 使用(这里还是自定义的)transform gpu利用率比预先除好tensor好效率更高. 可能是因为自带的torch totensor直接用比先弄好优化得好?

def train_dataset():
    return MyDataset(_trainImageFolder, _trainLabelPath, transform=_transform_train)


def val_dataset():
    return MyDataset(_valImageFolder, _valLabelPath, transform=_transform_val)


def cut_number():
    return eval(_trainLabelPath.split('\\')[-1].split('.')[0].split('_')[-1])


def data_name():
    return Path(__file__).name[:-3]
