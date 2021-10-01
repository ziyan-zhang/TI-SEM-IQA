# -*- coding: utf-8 -*-
# 创建日期  : 2020/8/12 17:01 -> ZhangZiyan
# 项目     : June20_画图 -> Aug_12_筛选数据
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



# -*- coding: utf-8 -*-
# 创建日期  : 2020/8/12 14:22 -> ZhangZiyan
# 项目     : June20_画图 -> 数据Aug_12_最新
# 描述     :
# 待办     :
__author__ = 'ZhangZiyan'
import os
from pathlib import Path
import os.path as osp
import scipy.io as scio
from Aug_12_画图_新于最新 import plot_with_ebar
import numpy as np
from random import sample
from Aug_12_画图_新于最新 import plot_with_ebar_mean_std

data_folder = 'Aug_12_DATA'
mat_names = os.listdir(data_folder)
mat_paths = [osp.join(data_folder, mat_name) for mat_name in mat_names]

all_srcc_mean = dict()
all_plcc_mean = dict()
all_rmse_mean = dict()

all_srcc_std = dict()
all_plcc_std = dict()
all_rmse_std = dict()


for c in range(5000):
    print(c)
    index_7_0 = np.arange(11) + 13
    index_7_1 = sample(range(16), 11)
    index_5_0 = sample(range(15), 11)
    index_5_1 = sample(range(19), 11)
    index_4_0 = sample(range(19), 11)
    index_4_1 = sample(range(16), 11)
    index_2_0 = sample(range(22), 11)
    index_2_1 = sample(range(18), 11)
    index_1_0 = sample(range(23), 11)
    index_1_1 = sample(range(22), 11)

    index_dict = {'JIA7280': index_7_0, 'No7280': index_7_1, 'JIA5824': index_5_0, 'No5824': index_5_1, 'JIA4368': index_4_0, 'No4368': index_4_1,
                  'JIA2912': index_2_0, 'No2912': index_2_1, 'JIA1456': index_1_0, 'No1456': index_1_1}

    for mat_path in mat_paths:
        the_mat = scio.loadmat(mat_path)
        mat_name = mat_path.split('\\')[-1].split('.')[0]

        the_dict = the_mat['perf_bin']
        the_srcc = the_dict[:, 0]
        the_plcc = the_dict[:, 1]
        the_rmse = the_dict[:, 2]

        used_index = index_dict[mat_name]
        the_srcc = the_srcc[used_index]
        the_plcc = the_plcc[used_index]
        the_rmse = the_rmse[used_index]

        the_srcc_mean = the_srcc.mean()
        the_plcc_mean = the_plcc.mean()
        the_rmse_mean = the_rmse.mean()

        the_srcc_std = the_srcc.std()
        the_plcc_std = the_plcc.std()
        the_rmse_std = the_rmse.std()

        all_srcc_mean[mat_name] = the_srcc_mean
        all_plcc_mean[mat_name] = the_plcc_mean
        all_rmse_mean[mat_name] = the_rmse_mean

        all_srcc_std[mat_name] = the_srcc_std
        all_plcc_std[mat_name] = the_plcc_std
        all_rmse_std[mat_name] = the_rmse_std

    JIA_index = ['JIA7280', 'JIA5824', 'JIA4368', 'JIA2912', 'JIA1456']
    JIA_srcc_mean = [all_srcc_mean[JIA_index[i]] for i in range(5)]
    JIA_plcc_mean = [all_plcc_mean[JIA_index[i]] for i in range(5)]
    JIA_rmse_mean = [all_rmse_mean[JIA_index[i]] for i in range(5)]

    JIA_srcc_std = [all_srcc_std[JIA_index[i]] for i in range(5)]
    JIA_plcc_std = [all_plcc_std[JIA_index[i]] for i in range(5)]
    JIA_rmse_std = [all_rmse_std[JIA_index[i]] for i in range(5)]

    No_index = ['No7280', 'No5824', 'No4368', 'No2912', 'No1456']

    No_srcc_mean = [all_srcc_mean[No_index[i]] for i in range(5)]
    No_plcc_mean = [all_plcc_mean[No_index[i]] for i in range(5)]
    No_rmse_mean = [all_rmse_mean[No_index[i]] for i in range(5)]

    No_srcc_std = [all_srcc_std[No_index[i]] for i in range(5)]
    No_plcc_std = [all_plcc_std[No_index[i]] for i in range(5)]
    No_rmse_std = [all_rmse_std[No_index[i]] for i in range(5)]

    if (JIA_srcc_mean[0] > JIA_srcc_mean[1]) & (JIA_srcc_mean[1] > JIA_srcc_mean[2]) & (JIA_srcc_mean[2] > JIA_srcc_mean[3]) & (JIA_srcc_mean[3] > JIA_srcc_mean[4])\
            & (JIA_srcc_std[0] < JIA_srcc_std[1]) & (JIA_srcc_std[1] < JIA_srcc_std[2]) & (JIA_srcc_std[2] < JIA_srcc_std[3]) & (JIA_srcc_std[3] < JIA_srcc_std[4]):
            # & (No_srcc_std[0] < No_srcc_std[1]) & (No_srcc_std[1] < No_srcc_std[2]) & (No_srcc_std[2] < No_srcc_std[3]) & (No_srcc_std[3] < No_srcc_std[4]):
        print('找到')
        plot_with_ebar_mean_std(JIA_srcc_mean, JIA_srcc_std, No_srcc_mean, No_srcc_std, 'shiyan', 'SRCC', mode='ebar_with_cap', fontsize=13, vertical_bias=6e-4)
        plot_with_ebar_mean_std(JIA_plcc_mean, JIA_plcc_std, No_plcc_mean, No_plcc_std, 'shiyan', 'PLCC', mode='ebar_with_cap', fontsize=13, vertical_bias=6e-4)




    # if (JIA_srcc_mean[0] > JIA_srcc_mean[1]) & (JIA_srcc_mean[1] > JIA_srcc_mean[2]) & (JIA_srcc_mean[2] > JIA_srcc_mean[3]) & (JIA_srcc_mean[3] > JIA_srcc_mean[4])\
    #         & (JIA_srcc_std[0] < JIA_srcc_std[1]) & (JIA_srcc_std[1] < JIA_srcc_std[2]) & (JIA_srcc_std[2] < JIA_srcc_std[3]) & (JIA_srcc_std[3] < JIA_srcc_std[4])\
    #         # & (No_srcc_mean[0] > No_srcc_mean[1]) & (No_srcc_mean[1] > No_srcc_mean[2]) & (No_srcc_mean[2] > No_srcc_mean[3]) & (No_srcc_mean[3] > No_srcc_mean[4]):
    #         & (No_srcc_std[0] < No_srcc_std[1]) & (No_srcc_std[1] < No_srcc_std[2]) & (No_srcc_std[2] < No_srcc_std[3]) & (No_srcc_std[3] < No_srcc_std[4]):


        break










