# -*- coding: utf-8 -*-
# 创建日期  : 2020/9/15 20:01 -> ZhangZiyan
# 项目     : June20_画图 -> 算cornia模糊表现
# 描述     :  
# 待办     :  
__author__ = 'ZhangZiyan'

import os
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
from shutil import copyfile
import time
# from progressbar import *
from scipy import stats
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import math
from PIL import Image

x_blur = []
y_blur = []
mat_path = 'srcc_0.9148_T0812_184804.mat'
mat = scio.loadmat(mat_path)
x = mat['pred_mean']
y = mat['y_mean']

x = x.squeeze()
y = y.squeeze()

for i in range(15):
    ith_batch = range(i*13+4, i*13+7)
    for ind in ith_batch:
        x_blur.append(x[ind].item())
        y_blur.append(y[ind].item())

