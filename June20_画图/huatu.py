"""-----------------------------------------------------
创建时间 :  2020/6/20  19:27
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

wc_t1_s = [0.906, 0.905, 0.912, 0.906, 0.907]
ck_t1_s = [0.905, 0.906, 0.904, 0.913]
pc_t1_s = [0.913, 0.913, 0.906]
t1_s = wc_t1_s + ck_t1_s + pc_t1_s
mt1_s = np.array(t1_s).mean()

wc_1_s = [0.894, 0.896, 0.892, 0.886]
ck_1_s = [0.892, 0.898]
pc_1_s = [0.894, 0.892, 0.901, 0.896]
n1 = wc_1_s + ck_1_s + pc_1_s
mn1 = np.array(n1).mean()

wc_t2_s = [0.906, 0.906, 0.905, 0.901, 0.905]
ck_t2_s = [0.911, 0.900, 0.898, 0.902]
pc_t2_s = [0.911, 0.898, 0.899]  # 这俩比较小的可以替换掉
# pc_t2_s = [0.911]

t2 = wc_t2_s + ck_t2_s + pc_t2_s
mt2 = np.array(t2).mean()

wc_2_s = [0.882, 0.890, 0.880, 0.887]
ck_2_s = [0.889, 0.884, 0.891, 0.886]
pc_2_s = [0.881, 0.893, 0.889]
n2 = wc_2_s + ck_2_s + pc_2_s
mn2 = np.array(n2).mean()

wc_t3_s = [0.903, 0.898, 0.910]
ck_t3_s = [0.907, 0.898, 0.905, 0.898, 0.898]
pc_t3_s = [0.907, 0.906, 0.901]
t3 = wc_t3_s + ck_t3_s + pc_t3_s
mt3 = np.array(t3).mean()

wc_3_s = [0.885, 0.878, 0.890, 0.885, 0.891]
ck_3_s = [0.876, 0.880, 0.881]
pc_3_s = [0.880, 0.891]
n3 = wc_3_s + ck_3_s + pc_3_s
mn3 = np.array(n3).mean()

wc_t4_s = [0.900, 0.905, 0.903, 0.901, 0.903]
ck_t4_s = [0.902, 0.889, 0.897]
pc_t4_s = [0.901, 0.899]
t4 = wc_t4_s + ck_t4_s + pc_t4_s
mt4 = np.array(t4).mean()

wc_4_s = [0.886, 0.882, 0.887, 0.878, 0.869]
ck_4_s = [0.874, 0.876, 0.887]
pc_4_s = [0.873, 0.882]
n4 = wc_4_s + ck_4_s + pc_4_s
mn4 = np.array(n4).mean()

# wc_t5 = [0.904, 0.904, 0.910, 0.905, 0.910]
# t5 = wc_t5
# mt5 = np.array(t5).mean()
#
# wc_5 = [0.885, 0.874, 0.882, 0.884, 0.881]
# n5 = wc_5
# mn5 = np.array(n5).mean()

# plotstyle = 'separate'
# plotstyle = 'together'
plotstyle = 'mean'

if plotstyle == 'separate':
    plt.plot(wc_t1_s, label='wc_t1_s', marker='x', linestyle='-', color='r')
    plt.plot(ck_t1_s, label='ck_t1_s', marker='|', linestyle='-', color='r')
    plt.plot(pc_t1_s, label='pc_t1_s', marker='o', linestyle='-', color='r')

    plt.plot(wc_1_s, label='wc_1_s', marker='x', linestyle='--', color='r')
    plt.plot(ck_1_s, label='ck_1_s', marker='|', linestyle='--', color='r')
    plt.plot(pc_1_s, label='pc_1_s', marker='o', linestyle='--', color='r')

    plt.plot(wc_t2_s, label='wc_t2_s', marker='x', linestyle='-', color='g')
    plt.plot(ck_t2_s, label='ck_t2_s', marker='x', linestyle='-', color='g')
    plt.plot(pc_t2_s, label='pc_t2_s', marker='x', linestyle='-', color='g')

    plt.plot(wc_2_s, label='wc_2_s', marker='x', linestyle='--', color='g')
    plt.plot(ck_2_s, label='ck_2_s', marker='x', linestyle='--', color='g')
    plt.plot(pc_2_s, label='pc_2_s', marker='x', linestyle='--', color='g')

    plt.plot(wc_t3_s, label='wc_t3_s', marker='x', linestyle='-', color='b')
    plt.plot(ck_t3_s, label='ck_t3_s', marker='x', linestyle='-', color='b')
    plt.plot(pc_t3_s, label='pc_t3_s', marker='x', linestyle='-', color='b')

    plt.plot(wc_3_s, label='wc_3_s', marker='x', linestyle='--', color='b')
    plt.plot(ck_3_s, label='ck_3_s', marker='x', linestyle='--', color='b')
    plt.plot(pc_3_s, label='pc_3_s', marker='x', linestyle='--', color='b')

    plt.plot(wc_t4_s, label='wc_t4_s', marker='x', linestyle='-', color='orange')

    plt.plot(wc_4_s, label='wc_4_s', marker='x', linestyle='--', color='orange')

    # plt.plot(wc_t5, label='wc_t5', marker='x', linestyle='-', color='gray')
    #
    # plt.plot(wc_5, label='wc_5', marker='x', linestyle='--', color='gray')

# 第二种画图方式
elif plotstyle == 'together':
    plt.plot(t1_s, label='t1_s', linestyle="-", color='r')
    plt.plot(n1, label='1', linestyle='--', color='r')

    plt.plot(t2, label='t2', linestyle="-", color='g')
    plt.plot(n2, label='2', linestyle='--', color='g')

    plt.plot(t3, label='t3', linestyle="-", color='b')
    plt.plot(n3, label='3', linestyle='--', color='b')

    plt.plot(t4, label='t4', linestyle="-", color='orange')
    plt.plot(n4, label='4', linestyle='--', color='orange')
    plt.text(0, 1, 'gg')

    plt.text(2, 0.9, "function: y = x * x", size=15, alpha=0.2)

elif plotstyle == 'mean':
    verticalBias = 6e-4
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(-0.2, 3.3)
    plt.ylim(0.875, 0.912)

    plt.plot([mt1_s, mt2, mt3, mt4], marker='o', label='proposed method', ls='-')
    plt.text(0, mt1_s+verticalBias, '%.3f' % mt1_s)
    plt.text(1, mt2+verticalBias, '%.3f' % mt2)
    plt.text(2, mt3+verticalBias, '%.3f' % mt3)
    plt.text(3, mt4+verticalBias, '%.3f' % mt4)

    plt.plot([mn1, mn2, mn3, mn4], marker='x', label='baseline', ls='--')
    plt.text(0, mn1+verticalBias, '%.3f' % mn1)
    plt.text(1, mn2+verticalBias, '%.3f' % mn2)
    plt.text(2, mn3+verticalBias, '%.3f' % mn3)
    plt.text(3, mn4+verticalBias, '%.3f' % mn4)

    x_ticks = ax.set_xticks([0, 1, 2, 3])
    xlabels = ax.set_xticklabels(['100%', '90%', '80%', '70%'], rotation=0)

    # ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='both')
    ax.set_xlabel('number of training samples')
    ax.set_ylabel('SRCC on the test set')

else:
    raise ValueError

plt.legend()
plt.savefig('performance drop.png', dpi=600)
plt.show()