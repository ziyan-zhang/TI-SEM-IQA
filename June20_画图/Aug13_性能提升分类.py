# -*- coding: utf-8 -*-
# 创建日期  : 2020/8/13 22:50 -> ZhangZiyan
# 项目     : June20_画图 -> Aug13_性能提升分类
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



"""-----------------------------------------------------
创建时间 :  2020/6/30  17:56
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from pathlib import Path
import time

def color(value):
  digit = list(map(str, range(10))) + list("ABCDEF")
  if isinstance(value, tuple):
    string = '#'
    for i in value:
      a1 = i // 16
      a2 = i % 16
      string += digit[a1] + digit[a2]
    return string
  elif isinstance(value, str):
    a1 = digit.index(value[1]) * 16 + digit.index(value[2])
    a2 = digit.index(value[3]) * 16 + digit.index(value[4])
    a3 = digit.index(value[5]) * 16 + digit.index(value[6])
    return (a1, a2, a3)


# sns.set(style='darkgrid')

label_list = ['All', 'Noise', 'Contrast', 'Blur', 'Brightness', 'Astigmation']

# srcc_propose = [0.9148, 0.8896, 0.8854, 0.8764, 0.8070, 0.7137]
# srcc_baseline = [0.8940, 0.8752, 0.8825, 0.8725, 0.8150, 0.6707]
# yLabel = "SRCC"


# 其实是plcc
srcc_propose = [0.8921, 0.8778, 0.8560, 0.8603, 0.8065, 0.8049]
srcc_baseline = [0.8863, 0.8626, 0.8161, 0.8513, 0.7878, 0.7867]
yLabel = "PLCC"

num_list3 = np.array(srcc_propose) - np.array(srcc_baseline)
x = range(len(srcc_propose))

verticalBias = 6e-4
Fontsize = 10
ALPHA = 1
WIDTH1 = 0.4
WIDTH = 0.45

colorA = color((31, 119, 180))
colorB = color((255, 127, 14))
plt.figure()
plt.grid(axis='y', ls='--')
ax = plt.gca()
ax.set_axisbelow(True)

rects1 = plt.bar(x=0-WIDTH1/2, height=srcc_propose[0], width=WIDTH1, alpha=ALPHA, label="proposed method")
rects1 = plt.bar(x=1-WIDTH1/2, height=srcc_propose[1], width=WIDTH1, alpha=ALPHA, color=colorA)  # )
rects1 = plt.bar(x=2-WIDTH1/2, height=srcc_propose[2], width=WIDTH1, alpha=ALPHA, color=colorA)
rects1 = plt.bar(x=3-WIDTH1/2, height=srcc_propose[3], width=WIDTH1, alpha=ALPHA, color=colorA)
rects1 = plt.bar(x=4-WIDTH1/2, height=srcc_propose[4], width=WIDTH1, alpha=ALPHA, color=colorA)
rects1 = plt.bar(x=5-WIDTH1/2, height=srcc_propose[5], width=WIDTH1, alpha=ALPHA, color=colorA)

plt.text(0-WIDTH, srcc_propose[0] + verticalBias, '%.3f' % srcc_propose[0], fontsize=Fontsize)
plt.text(1-WIDTH, srcc_propose[1] + verticalBias, '%.3f' % srcc_propose[1], fontsize=Fontsize)
plt.text(2-WIDTH, srcc_propose[2] + verticalBias, '%.3f' % srcc_propose[2], fontsize=Fontsize)
plt.text(3-WIDTH, srcc_propose[3] + verticalBias, '%.3f' % srcc_propose[3], fontsize=Fontsize)
plt.text(4-WIDTH, srcc_propose[4] + verticalBias, '%.3f' % srcc_propose[4], fontsize=Fontsize)
plt.text(5-WIDTH, srcc_propose[5] + verticalBias, '%.3f' % srcc_propose[5], fontsize=Fontsize)

rects2 = plt.bar(x=0+WIDTH1/2, height=srcc_baseline[0], width=WIDTH1, alpha=ALPHA,  label="only IQA network")
rects2 = plt.bar(x=1+WIDTH1/2, height=srcc_baseline[1], width=WIDTH1, alpha=ALPHA, color=colorB)  # ,color='gray'
rects2 = plt.bar(x=2+WIDTH1/2, height=srcc_baseline[2], width=WIDTH1, alpha=ALPHA, color=colorB)
rects2 = plt.bar(x=3+WIDTH1/2, height=srcc_baseline[3], width=WIDTH1, alpha=ALPHA, color=colorB)
rects2 = plt.bar(x=4+WIDTH1/2, height=srcc_baseline[4], width=WIDTH1, alpha=ALPHA, color=colorB)
rects2 = plt.bar(x=5+WIDTH1/2, height=srcc_baseline[5], width=WIDTH1, alpha=ALPHA, color=colorB)

plt.text(0, srcc_baseline[0] + verticalBias, '%.3f' % srcc_baseline[0], fontsize=Fontsize)
plt.text(1, srcc_baseline[1] + verticalBias, '%.3f' % srcc_baseline[1], fontsize=Fontsize)
plt.text(2, srcc_baseline[2] + verticalBias, '%.3f' % srcc_baseline[2], fontsize=Fontsize)
plt.text(3, srcc_baseline[3] + verticalBias, '%.3f' % srcc_baseline[3], fontsize=Fontsize)
plt.text(4, srcc_baseline[4] + verticalBias, '%.3f' % srcc_baseline[4], fontsize=Fontsize)
plt.text(5, srcc_baseline[5] + verticalBias, '%.3f' % srcc_baseline[5], fontsize=Fontsize)

plt.ylabel(yLabel, fontsize=12)
plt.xticks(x, label_list)
plt.xlabel("Types of Distortion", fontsize=12)

# PLCC
plt.ylim(0.77, 0.90)

# SRCC
# plt.ylim(0.632, 0.93)

plt.legend()

saveFolder = Path(__file__).name[:-3]
fileName = yLabel + time.strftime('%m%d_%H%M%S') + '.png'
plt.savefig(os.path.join(saveFolder, fileName), dpi=600, bbox_inches='tight')
plt.show()

