"""-----------------------------------------------------
创建时间 :  2020/05/25  21:02
todo    :
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, img_folder, labelFpath):
        super(MyDataset, self).__init__()
        self.img_names = [os.path.join(img_folder, fileName) for fileName in os.listdir(img_folder)]

        label_pd = pd.read_csv(labelFpath, header=None)
        label_array = np.array(label_pd)
        self.label_array = label_array.squeeze()

        self.img_reads = []
        for i in tqdm(range(len(self.img_names))):
            img_name_i = self.img_names[i]
            img_read_i = cv2.imread(img_name_i, 0)
            img_read_i = img_read_i.astype('float32')
            img_read_i = np.expand_dims(img_read_i, 0)
            img_read_i = img_read_i/255
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


def train_dataset():
    return MyDataset(_trainImageFolder, _trainLabelPath)


def val_dataset():
    return MyDataset(_valImageFolder, _valLabelPath)


def cut_number():
    return eval(_trainLabelPath.split('\\')[-1].split('.')[0].split('_')[-1])


def data_name():
    return Path(__file__).name[:-3]
