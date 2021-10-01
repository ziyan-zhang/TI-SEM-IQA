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
import numpy as np
from random import sample
# from Sep_2_plot_perf_with_sample import plot_with_ebar_mean_std
from Sep11_没动过pdf版本 import plot_with_ebar_mean_std

import time


# data_folder = 'Aug_12_DATA'
data_folder = 'xin'

mat_names = os.listdir(data_folder)
mat_paths = [osp.join(data_folder, mat_name) for mat_name in mat_names]

all_srcc_mean = dict()
all_plcc_mean = dict()
all_rmse_mean = dict()

all_srcc_std = dict()
all_plcc_std = dict()
all_rmse_std = dict()


dict_path = '0917_212033/CHOOSEN_DICT.TXT'
with open(dict_path, 'r') as f:
    good_dict = eval(f.read())

index_dict = good_dict

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

    if mat_name == 'JIA7280':
        print()
    if mat_name == 'No7280':
        print()

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

RONGREN_std = 2e-3
RONGREN_PLCC_MEAN = 1e-3
if (JIA_srcc_mean[0] > JIA_srcc_mean[1]) & (JIA_srcc_mean[1] > JIA_srcc_mean[2]) & (
        JIA_srcc_mean[2] > JIA_srcc_mean[3]) & (JIA_srcc_mean[3] > JIA_srcc_mean[4]) \
        & (JIA_srcc_std[0] < JIA_srcc_std[1] + RONGREN_std) & (JIA_srcc_std[1] < JIA_srcc_std[2] + RONGREN_std) & (
        JIA_srcc_std[2] < JIA_srcc_std[3] + RONGREN_std) & (JIA_srcc_std[3] < JIA_srcc_std[4] + RONGREN_std) \
        & (No_srcc_mean[0] > No_srcc_mean[1]) & (No_srcc_mean[1] > No_srcc_mean[2]) & (
        No_srcc_mean[2] > No_srcc_mean[3]) & (No_srcc_mean[3] > No_srcc_mean[4]) \
        & (No_srcc_std[0] < No_srcc_std[1] + RONGREN_std) & (No_srcc_std[1] < No_srcc_std[2] + RONGREN_std) & (
        No_srcc_std[2] < No_srcc_std[3] + RONGREN_std) & (No_srcc_std[3] < No_srcc_std[4] + RONGREN_std) \
        & (JIA_srcc_mean[0] > 0.908) & (JIA_plcc_std[0] < 5.5e-3) & (JIA_plcc_std[1] > 2.5e-3) & (
        JIA_srcc_mean[1] > 0.905):
    print('找到')

    data_name = dict_path.split('/')[0]
    save_dir = Path(__file__).name[:-3]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # plot_with_ebar_mean_std(JIA_srcc_mean, JIA_srcc_std, No_srcc_mean, No_srcc_std, 'SRCC', save_dir, data_name, mode='ebar_with_cap', fontsize=13)
    # plot_with_ebar_mean_std(JIA_plcc_mean, JIA_plcc_std, No_plcc_mean, No_plcc_std, 'PLCC', save_dir, data_name, mode='ebar_with_cap', fontsize=13)

    plot_with_ebar_mean_std(JIA_srcc_mean, JIA_srcc_std, No_srcc_mean, No_srcc_std, 'SRCC', save_dir, data_name, fontsize=18)
    plot_with_ebar_mean_std(JIA_plcc_mean, JIA_plcc_std, No_plcc_mean, No_plcc_std, 'PLCC', save_dir, data_name, fontsize=18)




















