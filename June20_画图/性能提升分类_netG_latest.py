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

sns.set(style='darkgrid')
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

# label_list = ['Blur', 'Contrast', 'Noise', 'Light', 'Astigmation']
# num_list1 = [0.8708, 0.8978, 0.8845, 0.8233, 0.7405]  # , 0.9130
# num_list1 = [0.8737, 0.9187, 0.8942, 0.8188, 0.7768]
# num_list2 = [0.8524, 0.8827, 0.8568, 0.8028, 0.6659]  #, 0.8957

label_list = ['Noise', 'Blur', 'Contrast', 'Light', 'Astigmation']
num_list1 = [0.8806, 0.8765, 0.8618, 0.8251, 0.7000]
num_list2 = [0.8595, 0.8656, 0.8671, 0.8039, 0.6830]

num_list3 = np.array(num_list1) - np.array(num_list2)
x = range(len(num_list1))

verticalBias = 6e-4

ALPHA = 1
WIDTH = 0.4
rects1 = plt.bar(x=[i-WIDTH/2 for i in x], height=num_list1, width=WIDTH, alpha=ALPHA, label="proposed", color='blue')
plt.text(0-WIDTH, num_list1[0] + verticalBias, '%.3f' % num_list1[0])
plt.text(1-WIDTH, num_list1[1] + verticalBias, '%.3f' % num_list1[1])
plt.text(2-WIDTH, num_list1[2] + verticalBias, '%.3f' % num_list1[2])
plt.text(3-WIDTH, num_list1[3] + verticalBias, '%.3f' % num_list1[3])
plt.text(4-WIDTH, num_list1[4] + verticalBias, '%.3f' % num_list1[4])
# plt.text(5-WIDTH, num_list1[5] + verticalBias, '%.3f' % num_list1[5])



rects2 = plt.bar(x=[i+WIDTH/2 for i in x], height=num_list2, width=WIDTH, alpha=ALPHA,  label="baseline", color='gray')
plt.text(0, num_list2[0] + verticalBias, '%.3f' % num_list2[0])
plt.text(1, num_list2[1] + verticalBias, '%.3f' % num_list2[1])
plt.text(2, num_list2[2] + verticalBias, '%.3f' % num_list2[2])
plt.text(3, num_list2[3] + verticalBias, '%.3f' % num_list2[3])
plt.text(4, num_list2[4] + verticalBias, '%.3f' % num_list2[4])
# plt.text(5, num_list2[5] + verticalBias, '%.3f' % num_list2[5])


plt.ylim(0.65, 0.9)
plt.ylabel("SRCC")
plt.xticks(x, label_list)
plt.xlabel("Types of Distortion")
# plt.title("某某公司")
plt.legend()
plt.savefig(Path(__file__).name[:-3] + time.strftime('%m%d_%H%M%S') + '.png', dpi=600, bbox_inches='tight')
plt.show()