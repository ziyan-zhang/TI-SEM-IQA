# -*- coding: utf-8 -*-
# 创建日期  : 2020/9/2 10:03 -> ZhangZiyan
# 项目     : June20_画图 -> Sep_2_plot_perf_with_sample
# 描述     :  
# 待办     :  
__author__ = 'ZhangZiyan'
import matplotlib.pyplot as plt
import seaborn
import time
import os

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


colorA = 'darkorange'
colorB = 'royalblue'


def plot_with_ebar_mean_std(mean_proposed, std_proposed, mean_baseline, std_baseline,  category, save_dir, data_name,
                            mode='ebar_with_cap', fontsize=13, horizontal_bias=0.05, vertical_bias=0):
    # sns.set_style(style='darkgrid')
    # fig, ax = plt.subplots()
    # x_ticks = [0, 1, 2, 3]
    x_ticks = [0, 1, 2, 3, 4]

    # plt.figure(num=None, figsize=(2.8, 1.7), dpi=300)
    plt.figure()
    plt.grid(axis='y', ls='--')
    ax = plt.gca()
    ax.set_axisbelow(True)

    if mode == 'ebar_without_cap':
        CapSize = 0
        plt.errorbar(x=x_ticks, y=mean_proposed, yerr=std_proposed, fmt='d-', color=colorA, ecolor=colorA,
                     label='proposed (weight transfer)', capsize=CapSize)
        plt.errorbar(x=x_ticks, y=mean_baseline, yerr=std_baseline, fmt='s-', color=colorB, ecolor=colorB,
                     label='only module2 (no weight transfer)', capsize=CapSize)
    elif mode == 'ebar_with_cap':
        CapSize = 3
        plt.errorbar(x=x_ticks, y=mean_proposed, yerr=std_proposed, fmt='d-', color=colorA, ecolor=colorA,
                     label='proposed method', capsize=CapSize)
        for x, y in zip(x_ticks, mean_proposed):
            plt.annotate("%.3f" % y, xy=(x+horizontal_bias, y+vertical_bias), bbox=dict(boxstyle='round,pad=0.05', fc=seaborn.xkcd_rgb['sandy yellow'], ec='k', lw=0.05, alpha=0.85),
                         fontsize=fontsize)

        plt.errorbar(x=x_ticks, y=mean_baseline, yerr=std_baseline, fmt='s-', color=colorB, ecolor=colorB,
                     label='only IQA network', capsize=CapSize)
        for x, y in zip(x_ticks, mean_baseline):
            plt.annotate("%.3f" % y, xy=(x+horizontal_bias, y+vertical_bias), bbox=dict(boxstyle='round,pad=0.05', fc=seaborn.xkcd_rgb['sandy yellow'], ec='k', lw=0.05, alpha=0.85),
                         fontsize=fontsize)
    elif mode == 'just plot':
        plt.plot(mean_proposed, marker='d', color=colorA, label='proposed')
        plt.plot(mean_baseline, marker='s', color=colorB, label='baseline')
    else:
        raise ValueError

    plt.xticks(x_ticks, rotation=0, fontproperties='Times New Roman', fontsize=fontsize)
    ax.set_xticklabels(['100%', '80%', '60%', '40%', '20%'])
    plt.yticks(rotation=0, fontproperties='Times New Roman', fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='best', prop={'family': 'Times New Roman', 'size': fontsize})

    plt.xlabel('Number of Training Samples', fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.ylabel(category, fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.xlim(-0.3, 4.55)
    plt.ylim(0.82, 0.92)
    plt.tight_layout()

    fig_name = data_name + category + mode + time.strftime('%m%d_%H%M%S') +'.png'
    plt.savefig(os.path.join(save_dir, data_name + '_' + fig_name), dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    mode1 = 'ebar_without_cap'
    mode2 = 'ebar_with_cap'
    mode3 = 'just plot'
    # plot_with_ebar(Proposed_Srcc_Mean, Proposed_Srcc_Error, Baseline_Srcc_Mean, Baseline_Srcc_Error, 'SRCC', mode1)
    # plot_with_ebar(Proposed_Plcc_Mean, Proposed_Plcc_Error, Baseline_Plcc_Mean, Baseline_Plcc_Error, 'PLCC', mode1)
    plot_with_ebar(Proposed_Srcc_Mean, Proposed_Srcc_Error, Baseline_Srcc_Mean, Baseline_Srcc_Error, 'SRCC', mode2)
    # plot_with_ebar(Proposed_Plcc_Mean, Proposed_Plcc_Error, Baseline_Plcc_Mean, Baseline_Plcc_Error, 'PLCC', mode2)
    # plot_with_ebar(Proposed_Srcc_Mean, Proposed_Srcc_Error, Baseline_Srcc_Mean, Baseline_Srcc_Error, 'SRCC', mode3)
    # plot_with_ebar(Proposed_Plcc_Mean, Proposed_Plcc_Error, Baseline_Plcc_Mean, Baseline_Plcc_Error, 'PLCC', mode3)






