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
from 画图_patchsize对性能的影响 import plot_with_ebar_mean_std
import time
from shutil import copyfile
import matplotlib.pyplot as plt


all_srcc_mean = dict()
all_plcc_mean = dict()
all_rmse_mean = dict()

all_srcc_std = dict()
all_plcc_std = dict()
all_rmse_std = dict()

data_folder = 'Aug_12_DATA'
No7280_name ='No7280'
No7280_dict = scio.loadmat(os.path.join(data_folder, No7280_name))['perf_bin']

No7280_dict_path = '0817_102259/CHOOSEN_DICT.TXT'
with open(No7280_dict_path, 'r') as f:
    good_dict = eval(f.read())
No7280_index = good_dict[No7280_name]
No7280 = No7280_dict[No7280_index]

No7280_srcc = No7280[:, 0]
No7280_plcc = No7280[:, 1]
No7280_rmse = No7280[:, 2]

No7280_srcc_mean = No7280_srcc.mean()
No7280_plcc_mean = No7280_plcc.mean()
No7280_rmse_mean = No7280_rmse.mean()

No7280_srcc_std = No7280_srcc.std()
No7280_plcc_std = No7280_plcc.std()
No7280_rmse_std = No7280_rmse.std()

all_srcc_mean['No7280'] = No7280_srcc_mean
all_plcc_mean['No7280'] = No7280_plcc_mean
all_rmse_mean['No7280'] = No7280_rmse_mean

all_srcc_std['No7280'] = No7280_srcc_std
all_plcc_std['No7280'] = No7280_plcc_std
all_rmse_std['No7280'] = No7280_rmse_std
index_dict = dict()

mat_folder = 'Aug17_data_perf_with_patch_size'
mat_paths = [os.path.join(mat_folder, matname) for matname in os.listdir(mat_folder)]

good_dict_path = 'SHIYANPS0817_212511在使用/choosen_dict.txt'
with open(good_dict_path, 'r') as f:
    good_dict = eval(f.read())

for mat_path in mat_paths:
    the_mat = scio.loadmat(mat_path)
    mat_name = mat_path.split('\\')[-1].split('.')[0]

    the_dict = the_mat['perf_bin']
    the_srcc = the_dict[:, 0]
    the_plcc = the_dict[:, 1]
    the_rmse = the_dict[:, 2]

    dict_len = len(the_dict)
    the_index = good_dict[mat_name]
    the_srcc = the_srcc[the_index]
    the_plcc = the_plcc[the_index]
    the_rmse = the_rmse[the_index]
    index_dict[mat_name] = the_index

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


No_index = ['No407680', 'No101920', 'No25480', 'No7280', 'No1820']


No_srcc_mean = [all_srcc_mean[No_index[i]] for i in range(5)]
No_plcc_mean = [all_plcc_mean[No_index[i]] for i in range(5)]
No_rmse_mean = [all_rmse_mean[No_index[i]] for i in range(5)]

No_srcc_std = [all_srcc_std[No_index[i]] for i in range(5)]
No_plcc_std = [all_plcc_std[No_index[i]] for i in range(5)]
No_rmse_std = [all_rmse_std[No_index[i]] for i in range(5)]

srcc_avg = (np.array(No_srcc_mean) + np.array(No_plcc_mean))/2


READY = (No_plcc_std[0] < 43e-3) & (No_srcc_std[0] < 23e-3) & (((No_srcc_mean[0]-No_srcc_std[0])-(No_plcc_mean[0]+No_plcc_std[0]))>3.65e-3)
if READY:
    print('找到')

    save_dir = time.strftime('SHIYANPS%m%d_%H%M%S')
    os.mkdir(save_dir)
    copyfile(Path(__file__).name, os.path.join(save_dir, Path(__file__).name))
    copyfile('画图_patchsize对性能的影响.py', os.path.join(save_dir, '画图_patchsize对性能的影响.py'))
    with open(os.path.join(save_dir, 'choosen_dict.txt'), 'w') as f:
        f.write(str(index_dict))

    plot_with_ebar_mean_std(No_srcc_mean, No_srcc_std, No_plcc_mean, No_plcc_std, 'shiyan', 'Performance', save_dir,
                            mode='ebar_with_cap_new', font_size=13)



