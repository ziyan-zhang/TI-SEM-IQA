"""-----------------------------------------------------
创建时间 :  2020/6/30  14:46
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-

import seaborn as sns
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

fig = plt.figure(dpi=100)
plt.rc('font', family='STFangsong')
sns.set(style="darkgrid")

# ============ 设置xlabel及ylabel ============
plt.xlim(102, 48)
x = np.linspace(100, 50, 6)
plt.xticks(x, fontsize=11)

plt.ylim(0, 1.04)
y = np.linspace(0, 1, 11)
plt.yticks(y, fontsize=11)

plt.xlabel('品质因子', fontdict={'color': 'black',
                             'family': 'STFangsong',
                             'weight': 'normal',
                             'size': 15})
plt.ylabel('F', fontdict={'color': 'black',
                          'fontstyle': 'italic',
                          'family': 'Times New Roman',
                          'weight': 'normal',
                          'size': 15})
# ================================

# ============ 显示数据 ============
x = np.linspace(100, 50, 6)
y = np.array([0.194173876, 0.161086478, 0.138896531, 0.129826697, 0.133716787, 0.152458326])

summary = []

for i in range(6):
    x_t = x[i]
    y_t = y[i]
    summary.append([x_t, y_t])

data = DataFrame(summary, columns=['品质因子', 'signal'])
# ================================

# 在图上绘制节点
sns.scatterplot(x="品质因子",
                y="signal",
                data=data)
# 在图上绘制线段
sns.lineplot(x="品质因子",
             y="signal",
             ci=None,
             data=data)

plt.show()