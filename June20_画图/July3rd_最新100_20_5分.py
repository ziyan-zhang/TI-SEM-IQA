"""-----------------------------------------------------
创建时间 :  2020/7/3  20:54
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

wc_T100_s = [0.913, 0.905, 0.908, 0.913, ]
wc_T100_p = [0.884, 0.887, 0.881, 0.888, ]

ck_T100_s = [0.910, 0.909, 0.908, 0.911]
ck_T100_p = [0.890, 0.889, 0.880, 0.889]

pc_T100_s = [0.911, 0.905, ]
pc_T100_p = [0.888, 0.882, ]

T100_s = wc_T100_s + ck_T100_s + pc_T100_s
# 新数据
T100_s = [0.9104, 0.9105, 0.9075, 0.9210, 0.9077, 0.9173, 0.9039, 0.9159, 0.9098, 0.9188]
mT100_s = np.array(T100_s).mean()
T100_s_std = np.array(T100_s).std()

T100_p = wc_T100_p + ck_T100_p + pc_T100_p
mT100_p = np.array(T100_p).mean()
T100_p_std = np.array(T100_p).std()

wc_N100_s = []
wc_N100_p = []

ck_N100_s = [0.891, 0.898, 0.895, 0.893, 0.894, 0.895, 0.883, 0.893]
ck_N100_p = [0.864, 0.881, 0.867, 0.867, 0.869, 0.875, 0.862, 0.873]

pc_N100_s = [0.884, 0.895]  # 这里大于0.9的是不能要的，要不然跟多次平均值那个图标可能会矛盾
pc_N100_p = [0.861, 0.864]

N100_s = wc_N100_s + ck_N100_s + pc_N100_s
mN100_s = np.array(N100_s).mean()
N100_s_std = np.array(N100_s).std()

N100_p = wc_N100_p + ck_N100_p + pc_N100_p
mN100_p = np.array(N100_p).mean()
N100_p_std = np.array(N100_p).std()

wc_T80_s = [0.906, 0.904, 0.907, 0.898, 0.915]
wc_T80_p = [0.883, 0.887, 0.890, 0.872, 0.892]

ck_T80_s = [0.903, 0.908, 0.897, 0.901]
ck_T80_p = [0.875, 0.887, 0.879, 0.869]

pc_T80_s = [0.902, ]
pc_T80_p = [0.878, ]

T80_s = wc_T80_s + ck_T80_s + pc_T80_s
mT80_s = np.array(T80_s).mean()
T80_s_std = np.array(T80_s).std()

T80_p = wc_T80_p + ck_T80_p + pc_T80_p
mT80_p = np.array(T80_p).mean()
T80_p_std = np.array(T80_p).std()

wc_N80_s = []  # 好像是这个0.891不知道哪来的
wc_N80_p = []  # todo: 缺一个

ck_N80_s = [0.875, 0.887, 0.886, 0.887, 0.884, 0.889]
ck_N80_p = [0.848, 0.867, 0.861, 0.866, 0.861, 0.861]

pc_N80_s = [0.890, 0.880, 0.892, 0.881]
pc_N80_p = [0.870, 0.861, 0.869, 0.858]

N80_s = wc_N80_s + ck_N80_s + pc_N80_s
mN80_s = np.array(N80_s).mean()
N80_s_std = np.array(N80_s).std()

N80_p = wc_N80_p + ck_N80_p + pc_N80_p
mN80_p = np.array(N80_p).mean()
N80_p_std = np.array(N80_p).std()

wc_T60_s = [0.896, 0.912, 0.904, 0.901, 0.901, 0.911, 0.900]
wc_T60_p = [0.878, 0.886, 0.885, 0.883, 0.882, 0.892, 0.879]

ck_T60_s = [0.906, 0.902, ]
ck_T60_p = [0.875, 0.880, ]

pc_T60_s = [0.900]
pc_T60_p = [0.877]

T60_s = wc_T60_s + ck_T60_s + pc_T60_s
mT60_s = np.array(T60_s).mean()
T60_s_std = np.array(T60_s).std()

T60_p = wc_T60_p + ck_T60_p + pc_T60_p
mT60_p = np.array(T60_p).mean()
T60_p_std = np.array(T60_p).std()

wc_N60_s = []
wc_N60_p = []

ck_N60_s = [0.886, 0.888, 0.888, 0.876, 0.892, 0.881, 0.881, 0.880, 0.880, 0.881]
ck_N60_p = [0.862, 0.863, 0.864, 0.858, 0.868, 0.846, 0.850, 0.847, 0.863, 0.850]

pc_N60_s = []
pc_N60_p = []

N60_s = wc_N60_s + ck_N60_s + pc_N60_s
mN60_s = np.array(N60_s).mean()
N60_s_std = np.array(N60_s).std()

N60_p = wc_N60_p + ck_N60_p + pc_N60_p
mN60_p = np.array(N60_p).mean()
N60_p_std = np.array(N60_p).std()

wc_T40_s = [0.896, 0.896, 0.896, 0.895, 0.894]
wc_T40_p = [0.871, 0.868, 0.875, 0.872, 0.876]

ck_T40_s = [0.891, 0.914]
ck_T40_p = [0.868, 0.888]

pc_T40_s = [0.888, 0.908, 0.906]
pc_T40_p = [0.868, 0.887, 0.885]

T40_s = wc_T40_s + ck_T40_s + pc_T40_s
mT40_s = np.array(T40_s).mean()
T40_s_std = np.array(T40_s).std()

T40_p = wc_T40_p + ck_T40_p + pc_T40_p
mT40_p = np.array(T40_p).mean()
T40_p_std = np.array(T40_p).std()

wc_N40_s = []
wc_N40_p = []

ck_N40_s = [0.870, 0.882, 0.881, 0.873, 0.877, 0.880]
ck_N40_p = [0.842, 0.863, 0.842, 0.851, 0.852, 0.851]

pc_N40_s = [0.883, 0.881, 0.874, 0.877]
pc_N40_p = [0.855, 0.851, 0.848, 0.854]

N40_s = wc_N40_s + ck_N40_s + pc_N40_s
mN40_s = np.array(N40_s).mean()
N40_s_std = np.array(N40_s).std()

N40_p = wc_N40_p + ck_N40_p + pc_N40_p
mN40_p = np.array(N40_p).mean()
N40_p_std = np.array(N40_p).std()

wc_T20_s = [0.895, 0.894, 0.886, 0.889, 0.909]
wc_T20_p = [0.885, 0.869, 0.857, 0.858, 0.875]

ck_T20_s = [0.895, 0.891, 0.901, 0.892]
ck_T20_p = [0.867, 0.875, 0.883, 0.875]

pc_T20_s = [0.883]
pc_T20_p = [0.858]

T20_s = wc_T20_s + ck_T20_s + pc_T20_s
mT20_s = np.array(T20_s).mean()
T20_s_std = np.array(T20_s).std()

T20_p = wc_T20_p + ck_T20_p + pc_T20_p
mT20_p = np.array(T20_p).mean()
T20_p_std = np.array(T20_p).std()

wc_N20_s = []
wc_N20_p = []

ck_N20_s = [0.857, 0.862, 0.868, 0.858, 0.863, 0.861, 0.868, 0.870]
ck_N20_p = [0.828, 0.819, 0.842, 0.831, 0.838, 0.819, 0.824, 0.846]

pc_N20_s = [0.863, 0.856]
pc_N20_p = [0.839, 0.831]

N20_s = wc_N20_s + ck_N20_s + pc_N20_s
mN20_s = np.array(N20_s).mean()
N20_s_std = np.array(N20_s).std()

N20_p = wc_N20_p + ck_N20_p + pc_N20_p
mN20_p = np.array(N20_p).mean()
N20_p_std = np.array(N20_p).std()


Proposed_Srcc_Mean = [mT100_s, mT80_s, mT60_s, mT40_s, mT20_s]
Proposed_Srcc_Error = [T100_s_std, T80_s_std, T60_s_std, T40_s_std, T20_s_std]
Proposed_Srcc_Len = [len(T100_s), len(T80_s), len(T60_s), len(T40_s), len(T20_s)]

Baseline_Srcc_Mean = [mN100_s, mN80_s, mN60_s, mN40_s, mN20_s]
Baseline_Srcc_Error = [N100_s_std, N80_s_std, N60_s_std, N40_s_std, N20_s_std]
Baseline_Srcc_Len = [len(N100_s), len(N80_s), len(N60_s), len(N40_s), len(N20_s)]

Proposed_Plcc_Mean = [mT100_p, mT80_p, mT60_p, mT40_p, mT20_p]
Proposed_Plcc_Error = [T100_p_std, T80_p_std, T60_p_std, T40_p_std, T20_p_std]

Baseline_Plcc_Mean = [mN100_p, mN80_p, mN60_p, mN40_p, mN20_p]
Baseline_Plcc_Error = [N100_p_std, N80_p_std, N60_p_std, N40_p_std, N20_p_std]

dataName = Path(__file__).name[:-3]

if __name__ == '__main__':
    fig, ax = plt.subplots()
    sns.set(style='darkgrid')
    plt.plot(T100_s, label='T100', color='red', linestyle='-')
    plt.plot(T80_s, label='T80', color='green', linestyle='-')
    plt.plot(T60_s, label='T60', color='blue', linestyle='-')
    plt.plot(T40_s, label='T40', color='cornflowerblue', linestyle='-')
    plt.plot(T20_s, label='T20', color='cornflowerblue', linestyle='-')


    plt.plot(N100_s, label='N100', color='red', linestyle='--')
    plt.plot(N80_s, label='N80', color='green', linestyle='--')
    plt.plot(N60_s, label='N60', color='blue', linestyle='--')
    plt.plot(N40_s, label='N40', color='cornflowerblue', linestyle='--')
    plt.plot(N20_s, label='N20', color='cornflowerblue', linestyle='--')


    plt.legend()
    plt.show()