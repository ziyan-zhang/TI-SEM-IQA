"""-----------------------------------------------------
创建时间 :  2020/6/24  17:18
说明    :
todo   :
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
"""-----------------------------------------------------
创建时间 :  2020/6/24  16:59
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

sns.set(style='darkgrid')
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

label_list = ['Blur', 'Contrast', 'Noise', 'Light', 'Astigmation']
num_list1 = [0.8833, 0.8811, 0.8520, 0.8144, 0.7143]  # , 0.9130
num_list2 = [0.8524, 0.8827, 0.8568, 0.8028, 0.6659]  #, 0.8957
num_list3 = np.array(num_list1) - np.array(num_list2)
x = range(len(num_list1))

verticalBias = 6e-4

ALPHA = 1
WIDTH = 0.4
rects1 = plt.bar(x=[i-WIDTH/2 for i in x], height=num_list1, width=WIDTH, alpha=ALPHA, label="proposed")
plt.text(0-WIDTH, num_list1[0] + verticalBias, '%.3f' % num_list1[0])
plt.text(1-WIDTH, num_list1[1] + verticalBias, '%.3f' % num_list1[1])
plt.text(2-WIDTH, num_list1[2] + verticalBias, '%.3f' % num_list1[2])
plt.text(3-WIDTH, num_list1[3] + verticalBias, '%.3f' % num_list1[3])
plt.text(4-WIDTH, num_list1[4] + verticalBias, '%.3f' % num_list1[4])
# plt.text(5-WIDTH, num_list1[5] + verticalBias, '%.3f' % num_list1[5])




rects2 = plt.bar(x=[i+WIDTH/2 for i in x], height=num_list2, width=WIDTH, alpha=ALPHA,  label="baseline")
plt.text(0, num_list2[0] + verticalBias, '%.3f' % num_list2[0])
plt.text(1, num_list2[1] + verticalBias, '%.3f' % num_list2[1])
plt.text(2, num_list2[2] + verticalBias, '%.3f' % num_list2[2])
plt.text(3, num_list2[3] + verticalBias, '%.3f' % num_list2[3])
plt.text(4, num_list2[4] + verticalBias, '%.3f' % num_list2[4])
# plt.text(5, num_list2[5] + verticalBias, '%.3f' % num_list2[5])



plt.ylim(0.6, 0.9)
plt.ylabel("SRCC")
plt.xticks(x, label_list)
plt.xlabel("Types of Distortion")
# plt.title("某某公司")
plt.legend()
plt.savefig('不同失真类型上的提升.png', dpi=600, bbox_inches='tight')
plt.show()