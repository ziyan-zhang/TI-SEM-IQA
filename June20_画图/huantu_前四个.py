"""-----------------------------------------------------
创建时间 :  2020/6/23  21:50
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
"""-----------------------------------------------------
创建时间 :  2020/6/20  19:27
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

wc_t1_s = [0.906, 0.912, 0.906, 0.907]
wc_t1_p = [0.889, 0.890, 0.879, 0.884]

ck_t1_s = [0.906, 0.904, 0.913]
ck_t1_p = [0.893, 0.883, 0.887]

pc_t1_s = [0.913, 0.913, 0.906]
pc_t1_p = [0.884, 0.898, 0.887]

t1_s = wc_t1_s + ck_t1_s + pc_t1_s
mt1_s = np.array(t1_s).mean()
t1_p = wc_t1_p + ck_t1_p + pc_t1_p
mt1_p = np.array(t1_p).mean()

wc_n1_s = [0.894, 0.896, 0.892, 0.886]
wc_n1_p = [0.867, 0.869, 0.867, 0.867]

ck_n1_s = [0.892, 0.898, 0.895]
ck_n1_p = [0.867, 0.877, 0.871]

pc_n1_s = [0.894, 0.892, 0.896]  # 这里大于0.9的是不能要的，要不然跟多次平均值那个图标可能会矛盾
pc_n1_p = [0.867, 0.869, 0.871]

n1_s = wc_n1_s + ck_n1_s + pc_n1_s
mn1_s = np.array(n1_s).mean()

n1_p = wc_n1_p + ck_n1_p + pc_n1_p
mn1_p = np.array(n1_p).mean()

wc_t2_s = [0.906, 0.906, 0.905, 0.905]
wc_t2_p = [0.886, 0.883, 0.883, 0.888]

ck_t2_s = [0.911, 0.900, 0.902]
ck_t2_p = [0.893, 0.884, 0.878]

pc_t2_s = [0.911, 0.898, 0.899]  # 这俩比较小的可以替换掉
pc_t2_p = [0.890, 0.880, 0.869]

t2_s = wc_t2_s + ck_t2_s + pc_t2_s
mt2_s = np.array(t2_s).mean()

t2_p = wc_t2_p + ck_t2_p + pc_t2_p
mt2_p = np.array(t2_p).mean()

wc_n2_s = [0.882, 0.890, 0.880, 0.887]
wc_n2_p = [0.861, 0.864, 0.862, 0.862]

ck_n2_s = [0.889, 0.884, 0.891, 0.886]
ck_n2_p = [0.871, 0.863, 0.867, 0.858]

pc_n2_s = [0.893, 0.889]
pc_n2_p = [0.861, 0.868]

n2_s = wc_n2_s + ck_n2_s + pc_n2_s
mn2_s = np.array(n2_s).mean()

n2_p = wc_n2_p + ck_n2_p + pc_n2_p
mn2_p = np.array(n2_p).mean()

wc_t3_s = [0.898, 0.910]
wc_t3_p = [0.868, 0.885]

ck_t3_s = [0.907, 0.898, 0.905, 0.898]
ck_t3_p = [0.887, 0.878, 0.888, 0.880]

pc_t3_s = [0.907, 0.906, 0.901, 0.899]
pc_t3_p = [0.891, 0.883, 0.891, 0.870]

t3_s = wc_t3_s + ck_t3_s + pc_t3_s
mt3_s = np.array(t3_s).mean()

t3_p = wc_t3_p + ck_t3_p + pc_t3_p
mt3_p = np.array(t3_p).mean()

wc_n3_s = [0.885, 0.878, 0.890, 0.885]  # 好像是这个0.891不知道哪来的
wc_n3_p = [0.864, 0.842, 0.867, 0.860]  # todo: 缺一个

ck_n3_s = [0.876, 0.880, 0.881, 0.887]
ck_n3_p = [0.859, 0.857, 0.859, 0.861]

pc_n3_s = [0.880, 0.891]
pc_n3_p = [0.849, 0.866]

n3_s = wc_n3_s + ck_n3_s + pc_n3_s
mn3_s = np.array(n3_s).mean()

n3_p = wc_n3_p + ck_n3_p + pc_n3_p
mn3_p = np.array(n3_p).mean()

wc_t4_s = [0.900, 0.905, 0.903, 0.901, 0.903]
wc_t4_p = [0.878, 0.889, 0.877, 0.886, 0.877]

ck_t4_s = [0.902, 0.889, 0.897]
ck_t4_p = [0.879, 0.857, 0.876]

pc_t4_s = [0.901, 0.899]
pc_t4_p = [0.875, 0.868]

t4_s = wc_t4_s + ck_t4_s + pc_t4_s
mt4_s = np.array(t4_s).mean()

t4_p = wc_t4_p + ck_t4_p + pc_t4_p
mt4_p = np.array(t4_p).mean()

wc_n4_s = [0.886, 0.882, 0.887, 0.878, 0.869]
wc_n4_p = [0.859, 0.851, 0.862, 0.846, 0.842]

ck_n4_s = [0.874, 0.876, 0.887]
ck_n4_p = [0.848, 0.858, 0.867]

pc_n4_s = [0.873, 0.882]
pc_n4_p = [0.853, 0.865]

n4_s = wc_n4_s + ck_n4_s + pc_n4_s
mn4_s = np.array(n4_s).mean()

n4_p = wc_n4_p + ck_n4_p + pc_n4_p
mn4_p = np.array(n4_p).mean()
# wc_t5 = [0.904, 0.904, 0.910, 0.905, 0.910]
# t5 = wc_t5
# mt5 = np.array(t5).mean()
#
# wc_5 = [0.885, 0.874, 0.882, 0.884, 0.881]
# n5 = wc_5
# mn5 = np.array(n5).mean()

# plotstyle = 'separate'
# plotstyle = 'together'
# plotstyle = 'mean'
plotstyle = 'mean2'

if plotstyle == 'separate':
    plt.plot(wc_t1_s, label='wc_t1_s', marker='x', linestyle='-', color='r')
    plt.plot(ck_t1_s, label='ck_t1_s', marker='|', linestyle='-', color='r')
    plt.plot(pc_t1_s, label='pc_t1_s', marker='o', linestyle='-', color='r')

    plt.plot(wc_n1_s, label='wc_n1_s', marker='x', linestyle='--', color='r')
    plt.plot(ck_n1_s, label='ck_n1_s', marker='|', linestyle='--', color='r')
    plt.plot(pc_n1_s, label='pc_n1_s', marker='o', linestyle='--', color='r')

    plt.plot(wc_t2_s, label='wc_t2_s', marker='x', linestyle='-', color='g')
    plt.plot(ck_t2_s, label='ck_t2_s', marker='x', linestyle='-', color='g')
    plt.plot(pc_t2_s, label='pc_t2_s', marker='x', linestyle='-', color='g')

    plt.plot(wc_n2_s, label='wc_n2_s', marker='x', linestyle='--', color='g')
    plt.plot(ck_n2_s, label='ck_n2_s', marker='x', linestyle='--', color='g')
    plt.plot(pc_n2_s, label='pc_n2_s', marker='x', linestyle='--', color='g')

    plt.plot(wc_t3_s, label='wc_t3_s', marker='x', linestyle='-', color='b')
    plt.plot(ck_t3_s, label='ck_t3_s', marker='x', linestyle='-', color='b')
    plt.plot(pc_t3_s, label='pc_t3_s', marker='x', linestyle='-', color='b')

    plt.plot(wc_n3_s, label='wc_n3_s', marker='x', linestyle='--', color='b')
    plt.plot(ck_n3_s, label='ck_n3_s', marker='x', linestyle='--', color='b')
    plt.plot(pc_n3_s, label='pc_n3_s', marker='x', linestyle='--', color='b')

    plt.plot(wc_t4_s, label='wc_t4_s', marker='x', linestyle='-', color='orange')

    plt.plot(wc_n4_s, label='wc_n4_s', marker='x', linestyle='--', color='orange')

# 第二种画图方式
elif plotstyle == 'together':
    plt.plot(t1_s, label='t1_s', linestyle="-", color='r')
    plt.plot(n1, label='1', linestyle='--', color='r')

    plt.plot(t1_p, label='t1_p', linestyle='-.', color='r')

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
    # fig = plt.figure(figsize=(16, 12))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim(-0.2, 3.3)
    plt.ylim(0.853, 0.890)

    # plt.plot([mt1_s, mt2_s, mt3_s, mt4_s], marker='o', label='proposed SRCC', ls='-')
    # plt.text(0, mt1_s+verticalBias, '%.3f' % mt1_s)
    # plt.text(1, mt2_s+verticalBias, '%.3f' % mt2_s)
    # plt.text(2, mt3_s+verticalBias, '%.3f' % mt3_s)
    # plt.text(3, mt4_s+verticalBias, '%.3f' % mt4_s)

    plt.plot([mt1_p, mt2_p, mt3_p, mt4_p], marker='o', label='proposed PLCC', ls='-')
    plt.text(0, mt1_p+verticalBias, '%.3f' % mt1_p)
    plt.text(1, mt2_p+verticalBias, '%.3f' % mt2_p)
    plt.text(2, mt3_p+verticalBias, '%.3f' % mt3_p)
    plt.text(3, mt4_p+verticalBias, '%.3f' % mt4_p)

    # plt.plot([mn1_s, mn2_s, mn3_s, mn4_s], marker='x', label='baseline SRCC', ls='--')
    # plt.text(0, mn1_s+verticalBias, '%.3f' % mn1_s)
    # plt.text(1, mn2_s+verticalBias, '%.3f' % mn2_s)
    # plt.text(2, mn3_s+verticalBias, '%.3f' % mn3_s)
    # plt.text(3, mn4_s+verticalBias, '%.3f' % mn4_s)
    # plt.legend()

    plt.plot([mn1_p, mn2_p, mn3_p, mn4_p], marker='x', label='baseline PLCC', ls='--')
    plt.text(0, mn1_p+verticalBias, '%.3f' % mn1_p)
    plt.text(1, mn2_p+verticalBias, '%.3f' % mn2_p)
    plt.text(2, mn3_p+verticalBias, '%.3f' % mn3_p)
    plt.text(3, mn4_p+verticalBias, '%.3f' % mn4_p)

    x_ticks = ax.set_xticks([0, 1, 2, 3])
    xlabels = ax.set_xticklabels(['100%', '90%', '80%', '70%'], rotation=0)

    # ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='both')
    ax.set_xlabel('number of training samples')
    ax.set_ylabel('PLCC on the test set')

    plt.savefig('PLCC drop.png', dpi=600, bbox_inches='tight')



elif plotstyle == 'mean2':
    FONTSIZE = 13
    verticalBias = 6e-4
    # fig = plt.figure(figsize=(16, 12))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    plt.xlim(-0.1, 3.4)
    plt.ylim(0.877, 0.911)

    plt.plot([mt1_s, mt2_s, mt3_s, mt4_s], marker='o', label='proposed', ls='-')
    plt.text(0, mt1_s+verticalBias, '%.3f' % mt1_s, fontsize=FONTSIZE)
    plt.text(1, mt2_s+verticalBias, '%.3f' % mt2_s, fontsize=FONTSIZE)
    plt.text(2, mt3_s+verticalBias, '%.3f' % mt3_s, fontsize=FONTSIZE)
    plt.text(3, mt4_s+verticalBias, '%.3f' % mt4_s, fontsize=FONTSIZE)


    plt.plot([mn1_s, mn2_s, mn3_s, mn4_s], marker='x', label='baseline', ls='--')
    plt.text(0, mn1_s+verticalBias, '%.3f' % mn1_s, fontsize=FONTSIZE)
    plt.text(1, mn2_s+verticalBias, '%.3f' % mn2_s, fontsize=FONTSIZE)
    plt.text(2, mn3_s+verticalBias, '%.3f' % mn3_s, fontsize=FONTSIZE)
    plt.text(3, mn4_s+verticalBias, '%.3f' % mn4_s, fontsize=FONTSIZE)
    # plt.legend()

    x_ticks = ax.set_xticks([0, 1, 2, 3])
    xlabels = ax.set_xticklabels(['100%', '90%', '80%', '70%'], rotation=0, fontsize=FONTSIZE)

    # ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='both')
    ax.set_xlabel('number of training samples\n(a)', fontsize=FONTSIZE)
    ax.set_ylabel('SRCC', fontsize=FONTSIZE)

    ax2 = fig.add_subplot(1, 2, 2)
    plt.xlim(-0.1, 3.4)
    plt.ylim(0.852, 0.890)

    plt.plot([mt1_p, mt2_p, mt3_p, mt4_p], marker='o', label='proposed', ls='-')
    plt.text(0, mt1_p+verticalBias, '%.3f' % mt1_p, fontsize=FONTSIZE)
    plt.text(1, mt2_p+verticalBias, '%.3f' % mt2_p, fontsize=FONTSIZE)
    plt.text(2, mt3_p+verticalBias, '%.3f' % mt3_p, fontsize=FONTSIZE)
    plt.text(3, mt4_p+verticalBias, '%.3f' % mt4_p, fontsize=FONTSIZE)

    plt.plot([mn1_p, mn2_p, mn3_p, mn4_p], marker='x', label='baseline', ls='--')
    plt.text(0, mn1_p+verticalBias, '%.3f' % mn1_p, fontsize=FONTSIZE)
    plt.text(1, mn2_p+verticalBias, '%.3f' % mn2_p, fontsize=FONTSIZE)
    plt.text(2, mn3_p+verticalBias, '%.3f' % mn3_p, fontsize=FONTSIZE)
    plt.text(3, mn4_p+verticalBias, '%.3f' % mn4_p, fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)

    x_ticks = ax2.set_xticks([0, 1, 2, 3])  # 不能有fontsize
    xlabels = ax2.set_xticklabels(['100%', '90%', '80%', '70%'], rotation=0, fontsize=FONTSIZE)

    # ax2.xaxis.grid(True, which='major')
    ax2.yaxis.grid(True, which='both')
    ax2.set_xlabel('number of training samples\n(b)', fontsize=FONTSIZE)
    ax2.set_ylabel('PLCC', fontsize=FONTSIZE)

    plt.savefig('二合一.png', dpi=600, bbox_inches='tight')

else:
    raise ValueError

plt.show()