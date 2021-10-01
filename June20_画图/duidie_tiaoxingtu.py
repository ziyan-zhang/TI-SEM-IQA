"""-----------------------------------------------------
创建时间 :  2020/6/24  16:59
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

label_list = ['blur', 'contrast', 'noise', 'light', 'others']
num_list1 = [0.8833, 0.8811, 0.8520, 0.8144, 0.7143]
num_list2 = [0.8524, 0.8827, 0.8568, 0.8028, 0.6659]
num_list3 = np.array(num_list1) - np.array(num_list2)
x = range(len(num_list1))
ALPHA = 0.7
rects1 = plt.bar(x=x, height=num_list1, width=0.45, alpha=ALPHA, color='blue', label="一部门")
rects2 = plt.bar(x=x, height=num_list3, width=0.45, alpha=ALPHA, color='yellow', label="二部门", bottom=num_list1)
plt.ylim(0.7, 0.9)
plt.ylabel("数量")
plt.xticks(x, label_list)
plt.xlabel("年份")
plt.title("某某公司")
plt.legend()
plt.show()