# -*- coding: utf-8 -*-
# 创建日期  : 2020/8/12 14:22 -> ZhangZiyan
# 项目     : June20_画图 -> 数据Aug_12_最新
# 描述     :  
# 待办     :  
__author__ = 'ZhangZiyan'
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from pathlib import Path
# from shutil import copyfile
# import time
# from progressbar import *
# from scipy import stats
import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# import pandas as pd
# import seaborn as sns
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.utils.data import sampler
# import math
import os.path as osp
import scipy.io as scio

from 画图纯手工 import plot_with_ebar


data_folder = 'Aug_12_DATA'
mat_names = os.listdir(data_folder)
mat_paths = [osp.join(data_folder, mat_name) for mat_name in mat_names]

all_srcc_dict = dict()
all_plcc_dict = dict()
all_rmse_dict = dict()

for mat_path in mat_paths:
    the_mat = scio.loadmat(mat_path)
    the_dict = the_mat['perf_bin']
    the_srcc = the_dict[:, 0]
    the_plcc = the_dict[:, 1]
    the_rmse = the_dict[:, 2]

    srcc_mean = the_srcc.mean()
    srcc_std = the_srcc.std()
    plcc_mean = the_plcc.mean()
    plcc_std = the_plcc.std()
    rmse_mean = the_rmse.mean()
    rmse_std = the_rmse.std()

    mat_name = mat_path.split('\\')[-1].split('.')[0]
    all_srcc_dict[mat_name] = np.array([srcc_mean, srcc_std])
    all_plcc_dict[mat_name] = np.array([plcc_mean, plcc_std])
    all_rmse_dict[mat_name] = np.array([rmse_mean, rmse_std])

JIA_index = ['JIA7280', 'JIA5824', 'JIA4368', 'JIA2912', 'JIA1456']
JIA_srcc = [all_srcc_dict[JIA_index[i]] for i in range(5)]
JIA_srcc = np.array(JIA_srcc)
JIA_plcc = [all_plcc_dict[JIA_index[i]] for i in range(5)]
JIA_plcc = np.array(JIA_plcc)
JIA_rmse = [all_rmse_dict[JIA_index[i]] for i in range(5)]
JIA_rmse = np.array(JIA_rmse)


No_index = ['No7280', 'No5824', 'No4368', 'No2912', 'No1456']
No_srcc = [all_srcc_dict[No_index[i]] for i in range(5)]
No_srcc = np.array(No_srcc)
No_plcc = [all_plcc_dict[No_index[i]] for i in range(5)]
No_plcc = np.array(No_plcc)
No_rmse = [all_rmse_dict[No_index[i]] for i in range(5)]
No_rmse = np.array(No_rmse)

plot_with_ebar(JIA_srcc[:, 0], JIA_srcc[:, 1], No_srcc[:, 0], No_srcc[:, 1], 'SRCC', 'ebar_with_cap')



