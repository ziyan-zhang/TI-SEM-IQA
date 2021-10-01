"""-----------------------------------------------------
创建时间 :  2020/5/8  16:05
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-

import math
import time
import os
import scipy
from scipy.signal import convolve2d
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import json
from my_utils.things_in_char import char_shijian
from my_utils.things_in_char import char_number
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
import cv2
from shutil import copy
from tensorboardX import SummaryWriter
from torchviz import make_dot
import shutil

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号, 注意['SimHei']对应这句不行.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = cv2.imread('E:\\SEMBig_Qianjing\\train_img\\01_001.tif', 0)
mask = cv2.imread('E:\\SEMBig_Qianjing\\train_seg\\01_001binary.png', 0)
mask_inverse = 1 - mask
# img_qianjing = img*mask
img_qianjing2 = cv2.bitwise_and(img, mask)
img_qianjing2_lightbg = img_qianjing2 + mask_inverse*255
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.imshow(img_qianjing2, cmap='gray')
plt.subplot(133)
plt.imshow(img_qianjing2_lightbg, cmap='gray')
outName = 'out.png'
cv2.imwrite(outName, img_qianjing2_lightbg)

