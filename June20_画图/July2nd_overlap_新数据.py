"""-----------------------------------------------------
创建时间 :  2020/7/2  22:30
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

wc_T100_s = [0.913, 0.905, 0.908, ]
wc_T100_p = [0.884, 0.887, 0.881, ]

ck_T100_s = []
ck_T100_p = []

pc_T100_s = [0.911, 0.905, ]
pc_T100_p = [0.888, 0.882, ]

T100_s = wc_T100_s + ck_T100_s + pc_T100_s
mT100_s = np.array(T100_s).mean()
T100_s_std = np.array(T100_s).std()

T100_p = wc_T100_p + ck_T100_p + pc_T100_p
mT100_p = np.array(T100_p).mean()
T100_p_std = np.array(T100_p).std()

wc_N100_s = []
wc_N100_p = []

ck_N100_s = [0.891, 0.898, ]
ck_N100_p = [0.864, 0.881, ]

pc_N100_s = []  # 这里大于0.9的是不能要的，要不然跟多次平均值那个图标可能会矛盾
pc_N100_p = []

N100_s = wc_N100_s + ck_N100_s + pc_N100_s
mN100_s = np.array(N100_s).mean()
N100_s_std = np.array(N100_s).std()

N100_p = wc_N100_p + ck_N100_p + pc_N100_p
mN100_p = np.array(N100_p).mean()
N100_p_std = np.array(N100_p).std()

wc_T90_s = [0.907, 0.903, ]
wc_T90_p = [0.886, 0.870, ]

ck_T90_s = []
ck_T90_p = []

pc_T90_s = [0.908, 0.905, ]  # 这俩比较小的可以替换掉
pc_T90_p = [0.890, 0.887,]

T90_s = wc_T90_s + ck_T90_s + pc_T90_s
mT90_s = np.array(T90_s).mean()
T90_s_std = np.array(T90_s).std()

T90_p = wc_T90_p + ck_T90_p + pc_T90_p
mT90_p = np.array(T90_p).mean()
T90_p_std = np.array(T90_p).std()

wc_N90_s = []
wc_N90_p = []

ck_N90_s = [0.884, 0.880, ]
ck_N90_p = [0.860, 0.857, ]

pc_N90_s = []
pc_N90_p = []

N90_s = wc_N90_s + ck_N90_s + pc_N90_s
mN90_s = np.array(N90_s).mean()
N90_s_std = np.array(N90_s).std()

N90_p = wc_N90_p + ck_N90_p + pc_N90_p
mN90_p = np.array(N90_p).mean()
N90_p_std = np.array(N90_p).std()

wc_T80_s = [0.906, 0.904, ]
wc_T80_p = [0.883, 0.887, ]

ck_T80_s = []
ck_T80_p = []

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

ck_N80_s = [0.875, ]
ck_N80_p = [0.848, ]

pc_N80_s = []
pc_N80_p = []

N80_s = wc_N80_s + ck_N80_s + pc_N80_s
mN80_s = np.array(N80_s).mean()
N80_s_std = np.array(N80_s).std()

N80_p = wc_N80_p + ck_N80_p + pc_N80_p
mN80_p = np.array(N80_p).mean()
N80_p_std = np.array(N80_p).std()

wc_T70_s = [0.907, 0.907, ]
wc_T70_p = [0.886, 0.882, ]

ck_T70_s = []
ck_T70_p = []

pc_T70_s = [0.904, ]
pc_T70_p = [0.889, ]

T70_s = wc_T70_s + ck_T70_s + pc_T70_s
mT70_s = np.array(T70_s).mean()
T70_s_std = np.array(T70_s).std()

T70_p = wc_T70_p + ck_T70_p + pc_T70_p
mT70_p = np.array(T70_p).mean()
T70_p_std = np.array(T70_p).std()

wc_N70_s = []
wc_N70_p = []

ck_N70_s = [0.881, ]
ck_N70_p = [0.853, ]

pc_N70_s = []
pc_N70_p = []

N70_s = wc_N70_s + ck_N70_s + pc_N70_s
mN70_s = np.array(N70_s).mean()
N70_s_std = np.array(N70_s).std()

N70_p = wc_N70_p + ck_N70_p + pc_N70_p
mN70_p = np.array(N70_p).mean()
N70_p_std = np.array(N70_p).std()

wc_T60_s = [0.896, 0.912, 0.904, ]
wc_T60_p = [0.878, 0.886, 0.885, ]

ck_T60_s = []
ck_T60_p = []

pc_T60_s = []
pc_T60_p = []

T60_s = wc_T60_s + ck_T60_s + pc_T60_s
mT60_s = np.array(T60_s).mean()
T60_s_std = np.array(T60_s).std()

T60_p = wc_T60_p + ck_T60_p + pc_T60_p
mT60_p = np.array(T60_p).mean()
T60_p_std = np.array(T60_p).std()

wc_N60_s = []
wc_N60_p = []

ck_N60_s = [0.886, ]
ck_N60_p = [0.862, ]

pc_N60_s = []
pc_N60_p = []

N60_s = wc_N60_s + ck_N60_s + pc_N60_s
mN60_s = np.array(N60_s).mean()
N60_s_std = np.array(N60_s).std()

N60_p = wc_N60_p + ck_N60_p + pc_N60_p
mN60_p = np.array(N60_p).mean()
N60_p_std = np.array(N60_p).std()

wc_T50_s = [0.910, 0.897, ]
wc_T50_p = [0.893, 0.868, ]

ck_T50_s = []
ck_T50_p = []

pc_T50_s = []
pc_T50_p = []

T50_s = wc_T50_s + ck_T50_s + pc_T50_s
mT50_s = np.array(T50_s).mean()
T50_s_std = np.array(T50_s).std()

T50_p = wc_T50_p + ck_T50_p + pc_T50_p
mT50_p = np.array(T50_p).mean()
T50_p_std = np.array(T50_p).std()

wc_N50_s = []
wc_N50_p = []

ck_N50_s = [0.886, ]
ck_N50_p = [0.862, ]

pc_N50_s = []
pc_N50_p = []

N50_s = wc_N50_s + ck_N50_s + pc_N50_s
mN50_s = np.array(N50_s).mean()
N50_s_std = np.array(N50_s).std()

N50_p = wc_N50_p + ck_N50_p + pc_N50_p
mN50_p = np.array(N50_p).mean()
N50_p_std = np.array(N50_p).std()


Proposed_Srcc_Mean = [mT100_s, mT90_s, mT80_s, mT70_s, mT60_s, mT50_s]
Proposed_Srcc_Error = [T100_s_std, T90_s_std, T80_s_std, T70_s_std, T60_s_std, T50_s_std]
Proposed_Srcc_Len = [len(T100_s), len(T90_s), len(T80_s), len(T70_s), len(T60_s), len(T50_s)]

Baseline_Srcc_Mean = [mN100_s, mN90_s, mN80_s, mN70_s, mN60_s, mN50_s]
Baseline_Srcc_Error = [N100_s_std, N90_s_std, N80_s_std, N70_s_std, N60_s_std, N50_s_std]
Baseline_Srcc_Len = [len(N100_s), len(N90_s), len(N80_s), len(N70_s), len(N60_s), len(N50_s)]

Proposed_Plcc_Mean = [mT100_p, mT90_p, mT80_p, mT70_p, mT60_p, mT50_p]
Proposed_Plcc_Error = [T100_p_std, T90_p_std, T80_p_std, T70_p_std, T60_p_std, T50_p_std]

Baseline_Plcc_Mean = [mN100_p, mN90_p, mN80_p, mN70_p, mN60_p, mN50_p]
Baseline_Plcc_Error = [N100_p_std, N90_p_std, N80_p_std, N70_p_std, N60_p_std, N50_p_std]


if __name__ == '__main__':
    fig, ax = plt.subplots()
    sns.set(style='darkgrid')
    plt.plot(T100_s, label='T100', color='red', linestyle='-')
    plt.plot(T90_s, label='T90', color='tomato', linestyle='-')
    plt.plot(T80_s, label='T80', color='green', linestyle='-')
    plt.plot(T70_s, label='T70', color='lightgreen', linestyle='-')
    plt.plot(T60_s, label='T60', color='blue', linestyle='-')
    plt.plot(T50_s, label='T50', color='cornflowerblue', linestyle='-')

    plt.plot(N100_s, label='N100', color='red', linestyle='--')
    plt.plot(N90_s, label='N90', color='tomato', linestyle='--')
    plt.plot(N80_s, label='N80', color='green', linestyle='--')
    plt.plot(N70_s, label='N70', color='lightgreen', linestyle='--')
    plt.plot(N60_s, label='N60', color='blue', linestyle='--')
    plt.plot(N50_s, label='N50', color='cornflowerblue', linestyle='--')

    plt.legend()
    plt.show()