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


data_folder = 'Aug_12_DATA'
mat_names = os.listdir(data_folder)
mat_paths = [osp.join(data_folder, mat_name) for mat_name in mat_names]

all_srcc_dict = dict()
all_plcc_dict = dict()
all_rmse_dict = dict()

# index_7_0 = np.arange(11) + 5
# index_7_1 = sample(range(16), 11)
# index_5_0 = sample(range(15), 11)
# index_5_1 = sample(range(19), 11)
# index_4_0 = sample(range(19), 11)
# index_4_1 = sample(range(16), 11)
# index_2_0 = sample(range(22), 11)
# index_2_1 = sample(range(18), 11)
# index_1_0 = sample(range(23), 11)
# index_1_1 = sample(range(22), 11)

index_7_0 = np.arange(11) + round((38 - 11) / 2) - 1
index_7_1 = np.arange(11) + round((16 - 11) / 2) - 1
index_5_0 = np.arange(11) + round((15 - 11) / 2) - 1
index_5_1 = np.arange(11) + round((19 - 11) / 2) - 1
index_4_0 = np.arange(11) + round((19 - 11) / 2) - 1
index_4_1 = np.arange(11) + round((16 - 11) / 2) - 1
index_2_0 = np.arange(11) + round((22 - 11) / 2) - 1
index_2_1 = np.arange(11) + round((18 - 11) / 2) - 1
index_1_0 = np.arange(11) + round((23 - 11) / 2) - 1
index_1_1 = np.arange(11) + round((22 - 11) / 2) - 1


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

    all_srcc_dict[mat_name] = the_srcc
    all_plcc_dict[mat_name] = the_plcc
    all_rmse_dict[mat_name] = the_rmse

JIA_index = ['JIA7280', 'JIA5824', 'JIA4368', 'JIA2912', 'JIA1456']
JIA_srcc = [all_srcc_dict[JIA_index[i]] for i in range(5)]
JIA_plcc = [all_plcc_dict[JIA_index[i]] for i in range(5)]
JIA_rmse = [all_rmse_dict[JIA_index[i]] for i in range(5)]

No_index = ['No7280', 'No5824', 'No4368', 'No2912', 'No1456']
No_srcc = [all_srcc_dict[No_index[i]] for i in range(5)]
No_plcc = [all_plcc_dict[No_index[i]] for i in range(5)]
No_rmse = [all_rmse_dict[No_index[i]] for i in range(5)]

plot_with_ebar(JIA_srcc, No_srcc, Path(__file__).name[:-3], 'SRCC', 'ebar_with_cap')
plot_with_ebar(JIA_plcc, No_plcc, Path(__file__).name[:-3], 'PLCC', 'ebar_with_cap')





