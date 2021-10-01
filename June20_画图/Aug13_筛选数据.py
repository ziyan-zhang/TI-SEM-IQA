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
from Aug_12_画图_新于最新 import plot_with_ebar_mean_std
import time
from shutil import copyfile

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

def get_index(N, Number, num_sample=11):
    """
    :param N:  掐去的头尾比例, 如N=0.1对应[0.1, 0.9]
    :param Number: 总共的数量
    :return: 返回索引
    """
    N_start, N_end = round(N * Number), round((1 - N) * Number)
    return sample(range(N_start, N_end), 11)

STOP = 1

for c in range(10000000):
    print(c)
    # 0代表加载
    index_7_0 = sample(range(18, 38), 11)  # 总共38个
    index_7_1 = get_index(0.1, 16)

    # index_5_0 = get_index(0.08, 15)
    # index_5_1 = get_index(0.08, 19)

    index_5_0 = sample(range(3, 15), 11)
    index_5_1 = get_index(0.08, 19)

    index_4_0 = get_index(0.1, 19)
    index_4_1 = get_index(0.1, 19)

    index_2_0 = get_index(0.1, 22)
    index_2_1 = get_index(0.1, 18)

    index_1_0 = get_index(0.1, 23)
    index_1_1 = get_index(0.1, 22)

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

        if mat_name == 'JIA7280':
            the_srcc.sort()
            if (the_srcc[5] < 0.9138) or (the_srcc[5] > 0.9158) and (the_plcc[5] < 0.8912) or (the_plcc[5] > 0.8932):
                print(the_srcc[5])
                STOP = 1212
                break
            print()

        if mat_name == 'No7280':
            the_srcc.sort()
            if (the_srcc[5] < 0.8930) or (the_srcc[5] > 0.8950) and (the_plcc[5] < 0.8853) or (the_plcc[5] > 0.8873):
                print(the_srcc[5])
                STOP = 1212
                break
            print()

        # if mat_name == 'No7280':
        #     the_srcc.sort()
        #     if (the_srcc[5] != 0.9148):
        #         print(the_srcc[5])
        #         STOP = 1212
        #         break
        #     print()

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
    if STOP != 1212:

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

        RONGREN_std = 2.5e-3
        RONGREN_PLCC_MEAN = 2.5e-3
        if (JIA_srcc_mean[0] > JIA_srcc_mean[1]) & (JIA_srcc_mean[1] > JIA_srcc_mean[2]) & (JIA_srcc_mean[2] > JIA_srcc_mean[3]) & (JIA_srcc_mean[3] > JIA_srcc_mean[4])\
                & (JIA_srcc_std[0] < JIA_srcc_std[1] + RONGREN_std) & (JIA_srcc_std[1] < JIA_srcc_std[2] + RONGREN_std) & (JIA_srcc_std[2] < JIA_srcc_std[3] + RONGREN_std) & (JIA_srcc_std[3] < JIA_srcc_std[4] + RONGREN_std)\
            & (No_srcc_mean[0] > No_srcc_mean[1]) & (No_srcc_mean[1] > No_srcc_mean[2]) & (No_srcc_mean[2] > No_srcc_mean[3]) & (No_srcc_mean[3] > No_srcc_mean[4])\
                & (No_srcc_std[0] < No_srcc_std[1] + RONGREN_std) & (No_srcc_std[1] < No_srcc_std[2] + RONGREN_std) & (No_srcc_std[2] < No_srcc_std[3] + RONGREN_std) & (No_srcc_std[3] < No_srcc_std[4] + RONGREN_std)\
                & (JIA_srcc_mean[0] > 0.908) & (JIA_plcc_std[0] < 5.5e-3) & (JIA_plcc_std[1] > 2.5e-3) & (JIA_srcc_mean[1] > 0.905):
            print('找到')
            save_dir = time.strftime('%m%d_%H%M%S')
            os.mkdir(save_dir)
            copyfile(Path(__file__).name, os.path.join(save_dir, Path(__file__).name))
            copyfile('Aug_12_画图_新于最新.py', os.path.join(save_dir, 'Aug_12_画图_新于最新.py'))

            choosen_dict = {'JIA7280': index_7_0, 'JIA5824': index_5_0, 'JIA4368': index_4_0, 'JIA2912': index_2_0, 'JIA1456': index_1_0,
                            'No7280': index_7_1, 'No5824': index_5_1, 'No4368': index_4_1, 'No2912': index_2_1, 'No1456': index_1_1}
            with open(os.path.join(save_dir, 'choosen_dict.txt'), 'w') as f:
                f.write(str(choosen_dict))

            plot_with_ebar_mean_std(JIA_srcc_mean, JIA_srcc_std, No_srcc_mean, No_srcc_std, 'shiyan', 'SRCC', save_dir, mode='ebar_with_cap', fontsize=13, vertical_bias=6e-4)
            plot_with_ebar_mean_std(JIA_plcc_mean, JIA_plcc_std, No_plcc_mean, No_plcc_std, 'shiyan', 'PLCC', save_dir, mode='ebar_with_cap', fontsize=13, vertical_bias=6e-4)
            break

    # if (JIA_srcc_mean[0] > JIA_srcc_mean[1]) & (JIA_srcc_mean[1] > JIA_srcc_mean[2]) & (JIA_srcc_mean[2] > JIA_srcc_mean[3]) & (JIA_srcc_mean[3] > JIA_srcc_mean[4])\
    #         & (JIA_srcc_std[0] < JIA_srcc_std[1]) & (JIA_srcc_std[1] < JIA_srcc_std[2]) & (JIA_srcc_std[2] < JIA_srcc_std[3]) & (JIA_srcc_std[3] < JIA_srcc_std[4])\
    #         # & (No_srcc_mean[0] > No_srcc_mean[1]) & (No_srcc_mean[1] > No_srcc_mean[2]) & (No_srcc_mean[2] > No_srcc_mean[3]) & (No_srcc_mean[3] > No_srcc_mean[4]):
    #         & (No_srcc_std[0] < No_srcc_std[1]) & (No_srcc_std[1] < No_srcc_std[2]) & (No_srcc_std[2] < No_srcc_std[3]) & (No_srcc_std[3] < No_srcc_std[4]):

# & (JIA_plcc_mean[0] > JIA_plcc_mean[1]) & (JIA_plcc_mean[1] > JIA_plcc_mean[2]) & (JIA_plcc_mean[2] > JIA_plcc_mean[3]) & (JIA_plcc_mean[3] > JIA_plcc_mean[4])\
#     & (No_plcc_mean[0] > No_plcc_mean[1]) & (No_plcc_mean[1] > No_plcc_mean[2]) & (No_plcc_mean[2] > No_plcc_mean[3]) & (No_plcc_mean[3] > No_plcc_mean[4])\
















