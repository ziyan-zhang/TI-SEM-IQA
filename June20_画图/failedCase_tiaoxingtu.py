"""-----------------------------------------------------
创建时间 :  2020/6/24  11:48
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

label_list = ['blur', 'contrast', 'noise', 'brightness', 'astigmatism']    # 横坐标刻度显示值
num_list1 = [0.8833, 0.8811, 0.8520, 0.8144, 0.7143]      # 纵坐标值1
num_list2 = [0.8495, 0.8513, 0.8532, 0.8348, 0.7991]      # 纵坐标值2

x = range(len(num_list1))
"""
绘制条形图
left:长条形中点横坐标
height:长条形高度
width:长条形宽度，默认值0.8
label:为后面设置legend准备
"""
rects1 = plt.bar(x=x, height=num_list1, width=0.4, alpha=0.8, color='red', label="一部门")
rects2 = plt.bar(x=[i + 0.4 for i in x], height=num_list2, width=0.4, color='green', label="二部门")
plt.ylim(0.7, 0.9)     # y轴取值范围
plt.ylabel("SRCC/PLCC")
"""
设置x轴刻度显示值
参数一：中点坐标
参数二：显示值
"""
plt.xticks([index + 0.2 for index in x], label_list)
plt.xlabel("different kinds of distortions")
plt.title("performance")
plt.legend()     # 设置题注
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
plt.show()