# -*- coding: utf-8 -*-
# 创建日期  : 2020/8/13 16:05 -> ZhangZiyan
# 项目     : June20_画图 -> 演草
# 描述     :  
# 待办     :  
__author__ = 'ZhangZiyan'
import os
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
from shutil import copyfile
import time
from progressbar import *
from scipy import stats
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import math

the_mat = scio.loadmat('E:\June20_画图\Aug_12_DATA\JIA7280.mat')
the_mat = the_mat['perf_bin']
JIA7280 = np.array([10, 17, 19, 25, 24, 16, 21, 30, 23, 22, 20])
THEARRAY = the_mat[JIA7280]
the_ay_srcc = THEARRAY[:, 0]
np.argsort(the_ay_srcc)



