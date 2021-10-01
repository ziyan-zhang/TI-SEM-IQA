"""-----------------------------------------------------
创建时间 :  2020/7/1  19:39
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import numpy as np


wc_t1_s = [0.913, 0.908, 0.906, 0.913, 0.910, 0.909, 0.911]
wc_t1_p = [0.894, 0.887, 0.879, 0.888, 0.887, 0.882, 0.880]

ck_t1_s = [0.912, 0.910]
ck_t1_p = [0.884, 0.880]

pc_t1_s = [0.908, 0.903, 0.910]
pc_t1_p = [0.886, 0.877, 0.889]

t1_s = wc_t1_s + ck_t1_s + pc_t1_s
mt1_s = np.array(t1_s).mean()
t1_s_std = np.array(t1_s).std()

t1_p = wc_t1_p + ck_t1_p + pc_t1_p
mt1_p = np.array(t1_p).mean()
t1_p_std = np.array(t1_p).std()

wc_n1_s = [0.894, 0.896, 0.892]
wc_n1_p = [0.867, 0.869, 0.867]

ck_n1_s = [0.892, 0.898, 0.895]
ck_n1_p = [0.867, 0.877, 0.871]

pc_n1_s = [0.894, 0.892, 0.896, 0.898]  # 这里大于0.9的是不能要的，要不然跟多次平均值那个图标可能会矛盾
pc_n1_p = [0.867, 0.869, 0.871, 0.870]

n1_s = wc_n1_s + ck_n1_s + pc_n1_s
mn1_s = np.array(n1_s).mean()
n1_s_std = np.array(n1_s).std()

n1_p = wc_n1_p + ck_n1_p + pc_n1_p
mn1_p = np.array(n1_p).mean()
n1_p_std = np.array(n1_p).std()

wc_t2_s = [0.906] + [0.906, 0.908]
wc_t2_p = [0.888] + [0.877, 0.886]

ck_t2_s = [0.900, 0.909, 0.904, 0.900] + [0.904]
ck_t2_p = [0.873, 0.889, 0.882, 0.881] + [0.881]

pc_t2_s = [0.908, 0.903, 0.904] + [0.902, 0.901]  # 这俩比较小的可以替换掉
pc_t2_p = [0.883, 0.880, 0.885] + [0.874, 0.879]

t2_s = wc_t2_s + ck_t2_s + pc_t2_s
mt2_s = np.array(t2_s).mean()
t2_s_std = np.array(t2_s).std()

t2_p = wc_t2_p + ck_t2_p + pc_t2_p
mt2_p = np.array(t2_p).mean()
t2_p_std = np.array(t2_p).std()

wc_n2_s = [0.882, 0.890, 0.880, 0.887]
wc_n2_p = [0.861, 0.864, 0.862, 0.862]

ck_n2_s = [0.889, 0.884, 0.891, 0.886]
ck_n2_p = [0.871, 0.863, 0.867, 0.858]

pc_n2_s = [0.893, 0.889]
pc_n2_p = [0.861, 0.868]

n2_s = wc_n2_s + ck_n2_s + pc_n2_s
mn2_s = np.array(n2_s).mean()
n2_s_std = np.array(n2_s).std()

n2_p = wc_n2_p + ck_n2_p + pc_n2_p
mn2_p = np.array(n2_p).mean()
n2_p_std = np.array(n2_p).std()

wc_t3_s = [0.903, 0.901, 0.897, 0.898, 0.901] + [0.892, 0.897, 0.897, 0.907, 0.903, 0.902, 0.902, 0.906]
wc_t3_p = [0.875, 0.878, 0.880, 0.885, 0.881] + [0.871, 0.879, 0.876, 0.888, 0.878, 0.879, 0.884, 0.871]

ck_t3_s = [0.897, 0.899, 0.899] + [0.899, 0.905, 0.906, 0.893]
ck_t3_p = [0.875, 0.874, 0.881] + [0.866, 0.877, 0.877, 0.877]

pc_t3_s = [0.904, 0.900, 0.893] + [0.905, 0.902, 0.904]
pc_t3_p = [0.889, 0.878, 0.873] + [0.878, 0.880, 0.885]

t3_s = wc_t3_s + ck_t3_s + pc_t3_s
mt3_s = np.array(t3_s).mean()
t3_s_std = np.array(t3_s).std()

t3_p = wc_t3_p + ck_t3_p + pc_t3_p
mt3_p = np.array(t3_p).mean()
t3_p_std = np.array(t3_p).std()

wc_n3_s = [0.885, 0.878, 0.890, 0.885]  # 好像是这个0.891不知道哪来的
wc_n3_p = [0.864, 0.842, 0.867, 0.860]  # todo: 缺一个

ck_n3_s = [0.876, 0.880, 0.881, 0.887]
ck_n3_p = [0.859, 0.857, 0.859, 0.861]

pc_n3_s = [0.880, 0.891]
pc_n3_p = [0.849, 0.866]

n3_s = wc_n3_s + ck_n3_s + pc_n3_s
mn3_s = np.array(n3_s).mean()
n3_s_std = np.array(n3_s).std()

n3_p = wc_n3_p + ck_n3_p + pc_n3_p
mn3_p = np.array(n3_p).mean()
n3_p_std = np.array(n3_p).std()

wc_t4_s = [0.895, 0.893, 0.904, 0.902, 0.896, 0.892, 0.896, 0.904, 0.895, 0.904] + [0.898, 0.903, 0.897, 0.891, 0.904, 0.897]
wc_t4_p = [0.874, 0.871, 0.887, 0.880, 0.863, 0.870, 0.874, 0.885, 0.868, 0.877] + [0.875, 0.873, 0.876, 0.877, 0.888, 0.873]

ck_t4_s = [0.898, 0.900, 0.902, 0.905, 0.894, 0.898]
ck_t4_p = [0.878, 0.867, 0.884, 0.887, 0.870, 0.882]

pc_t4_s = [0.901] + [0.896, 0.907, 0.896]
pc_t4_p = [0.882] + [0.874, 0.887, 0.870]

t4_s = wc_t4_s + ck_t4_s + pc_t4_s
mt4_s = np.array(t4_s).mean()
t4_s_std = np.array(t4_s).std()

t4_p = wc_t4_p + ck_t4_p + pc_t4_p
mt4_p = np.array(t4_p).mean()
t4_p_std = np.array(t4_p).std()

wc_n4_s = [0.886, 0.882, 0.887, 0.878, 0.869]
wc_n4_p = [0.859, 0.851, 0.862, 0.846, 0.842]

ck_n4_s = [0.874, 0.876, 0.887]
ck_n4_p = [0.848, 0.858, 0.867]

pc_n4_s = [0.873, 0.882]
pc_n4_p = [0.853, 0.865]

n4_s = wc_n4_s + ck_n4_s + pc_n4_s
mn4_s = np.array(n4_s).mean()
n4_s_std = np.array(n4_s).std()

n4_p = wc_n4_p + ck_n4_p + pc_n4_p
mn4_p = np.array(n4_p).mean()
n4_p_std = np.array(n4_p).std()

Proposed_Srcc_Mean = [mt1_s, mt2_s, mt3_s, mt4_s]
Proposed_Srcc_Error = [t1_s_std, t2_s_std, t3_s_std, t4_s_std]
Proposed_Srcc_Len = [len(t1_s), len(t2_s), len(t3_s), len(t4_s)]

Baseline_Srcc_Mean = [mn1_s, mn2_s, mn3_s, mn4_s]
Baseline_Srcc_Error = [n1_s_std, n2_s_std, n3_s_std, n4_s_std]
Baseline_Srcc_Len = [len(n1_s), len(n2_s), len(n3_s), len(n4_s)]

Proposed_Plcc_Mean = [mt1_p, mt2_p, mt3_p, mt4_p]
Proposed_Plcc_Error = [t1_p_std, t2_p_std, t3_p_std, t4_p_std]

Baseline_Plcc_Mean = [mn1_p, mn2_p, mn3_p, mn4_p]
Baseline_Plcc_Error = [n1_p_std, n2_p_std, n3_p_std, n4_p_std]
