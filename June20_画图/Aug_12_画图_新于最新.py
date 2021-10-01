"""-----------------------------------------------------
创建时间 :  2020/7/1  14:45
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from matplotlib.pyplot import MultipleLocator
import os
import numpy as np
import seaborn



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


colorA = color((31, 119, 180))
colorB = color((255, 127, 14))



# def plot_with_ebar(perf_jia, perf_no, data_name, category, mode='ebar_with_cap', fontsize=13, vertical_bias=6e-4):
#     sns.set_style(style='darkgrid')
#     fig, ax = plt.subplots()
#     # x_ticks = [0, 1, 2, 3]
#     x_ticks = [0, 1, 2, 3, 4]
#
#     mean_proposed = [perf_jia[i].mean() for i in range(5)]
#     std_proposed = [perf_jia[i].std() for i in range(5)]
#     mean_baseline = [perf_no[i].mean() for i in range(5)]
#     std_baseline = [perf_no[i].std() for i in range(5)]
#
#     if mode == 'ebar_without_cap':
#         CapSize = 0
#         plt.errorbar(x=x_ticks, y=mean_proposed, yerr=std_proposed, fmt='d-', color=colorA, ecolor=colorA,
#                      label='proposed', capsize=CapSize)
#         plt.errorbar(x=x_ticks, y=mean_baseline, yerr=std_baseline, fmt='s-', color=colorB, ecolor=colorB,
#                      label='baseline', capsize=CapSize)
#     elif mode == 'ebar_with_cap':
#         CapSize = 3
#         plt.errorbar(x=x_ticks, y=mean_proposed, yerr=std_proposed, fmt='d-', color=colorA, ecolor=colorA,
#                      label='proposed', capsize=CapSize)
#         plt.errorbar(x=x_ticks, y=mean_baseline, yerr=std_baseline, fmt='s-', color=colorB, ecolor=colorB,
#                      label='baseline', capsize=CapSize)
#     elif mode == 'just plot':
#         plt.plot(mean_proposed, marker='d', color=colorA, label='proposed')
#         plt.plot(mean_baseline, marker='s', color=colorB, label='baseline')
#     else:
#         raise ValueError
#
#     for i in range(len(mean_proposed)):
#         plt.text(i, mean_proposed[i] + vertical_bias, '%.4f_' % mean_proposed[i]+str(len(perf_jia[i])), fontsize=fontsize)
#         plt.text(i, mean_baseline[i] + vertical_bias, '%.4f_' % mean_baseline[i]+str(len(perf_no[i])), fontsize=fontsize)
#
#     x_ticks = ax.set_xticks(x_ticks)
#     xlabels = ax.set_xticklabels(['100%', '80%', '60%', '40%', '20%'], rotation=0, fontsize=fontsize)
#
#     y_major_locator = MultipleLocator(0.005)
#     ax.yaxis.set_major_locator(y_major_locator)
#
#     # ax.yaxis.label.set_size(fontsize)  # 不管用
#
#     ax.set_xlabel('number of training samples', fontsize=fontsize)
#     ax.set_ylabel(category, fontsize=fontsize)
#     plt.xlim(-0.1, 4.4)
#     if mode == 'ebar_with_cap' or mode == 'ebar_without_cap':
#         # plt.ylim(mean_baseline[-1] - std_baseline[-1] - 5e-4, mean_proposed[0] + std_proposed[0] + 5e-4)
#         pass
#     plt.legend(fontsize=fontsize)
#
#     plt.tight_layout()
#     figname = Path(__file__).name[:-3] + data_name + category + mode + time.strftime('%m%d_%H%M%S') +'.png'
#     plt.savefig(figname, dpi=600, bbox_inches='tight')
#     plt.show()


def plot_with_ebar_mean_std(mean_proposed, std_proposed, mean_baseline, std_baseline, data_name, category, save_dir, mode='ebar_with_cap', fontsize=13, horizontal_bias=0.05, vertical_bias=0):
    # sns.set_style(style='darkgrid')
    # fig, ax = plt.subplots()
    # x_ticks = [0, 1, 2, 3]
    x_ticks = [0, 1, 2, 3, 4]

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

    # for i in range(len(mean_proposed)):
    #     plt.text(i, mean_proposed[i] + vertical_bias, '%.3f' % mean_proposed[i], fontsize=fontsize)
    #     plt.text(i, mean_baseline[i] + vertical_bias, '%.3f' % mean_baseline[i], fontsize=fontsize)

    x_ticks = ax.set_xticks(x_ticks)
    xlabels = ax.set_xticklabels(['100%', '80%', '60%', '40%', '20%'], rotation=0, fontsize=fontsize)

    # y_major_locator = MultipleLocator(0.005)
    # ax.yaxis.set_major_locator(y_major_locator)

    # ax.yaxis.label.set_size(fontsize)  # 不管用

    ax.set_xlabel('Number of Training Samples', fontsize=fontsize+2)
    ax.set_ylabel(category, fontsize=fontsize+2)
    plt.xlim(-0.3, 4.45)
    plt.ylim(0.82, 0.92)
    if mode == 'ebar_with_cap' or mode == 'ebar_without_cap':
        # plt.ylim(mean_baseline[-1] - std_baseline[-1] - 5e-4, mean_proposed[0] + std_proposed[0] + 5e-4)
        pass
    plt.legend(fontsize=fontsize, loc='best')

    plt.tight_layout()

    figname = data_name + category + mode + time.strftime('%m%d_%H%M%S') +'.png'
    plt.savefig(os.path.join(save_dir, figname), dpi=600, bbox_inches='tight')
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





