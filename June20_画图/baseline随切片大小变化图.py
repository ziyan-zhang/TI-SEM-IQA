# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
from July6th_切片大小性能 import SRCC_Mean, SRCC_Error, PLCC_Mean, PLCC_Error, SRCC_Len, PLCC_Len, dataName
from pathlib import Path
from matplotlib.pyplot import MultipleLocator
import numpy as np


def plot_with_ebar(mean_SRCC, error_SRCC, mean_PLCC, error_PLCC, mode='ebar_with_cap', fontsize=13, vertical_bias=6e-4):
    sns.set_style(style='darkgrid')
    fig, ax = plt.subplots()
    # x_ticks = [0, 1, 2, 3]
    x_ticks = [0, 1, 2, 3, 4]

    if mode == 'ebar_without_cap':
        CapSize = 0
        plt.errorbar(x=x_ticks, y=mean_SRCC, yerr=error_SRCC, fmt='s-', label='SRCC', capsize=CapSize)
        plt.errorbar(x=x_ticks, y=mean_PLCC, yerr=error_PLCC, fmt='s-', label='PLCC', capsize=CapSize)
    elif mode == 'ebar_with_cap':
        CapSize = 3
        plt.errorbar(x=x_ticks, y=mean_SRCC, yerr=error_SRCC, fmt='s-', label='SRCC', capsize=CapSize)
        plt.errorbar(x=x_ticks, y=mean_PLCC, yerr=error_PLCC, fmt='s-', label='PLCC', capsize=CapSize)
    elif mode == 'just plot':
        plt.plot(mean_SRCC, marker='s', label='SRCC')
        plt.plot(mean_PLCC, marker='s', label='PLCC')

    else:
        raise ValueError

    for i in range(len(mean_SRCC)):
        plt.text(i, mean_SRCC[i] + vertical_bias, '%.3f_' % mean_SRCC[i]+str(SRCC_Len[i]), fontsize=fontsize)
        plt.text(i, mean_PLCC[i] + vertical_bias, '%.3f_' % mean_PLCC[i]+str(PLCC_Len[i]), fontsize=fontsize)


    x_ticks = ax.set_xticks(x_ticks)
    xlabels = ax.set_xticklabels(['512', '256', '128', '64', '32'], rotation=0, fontsize=fontsize)

    y_major_locator = MultipleLocator(0.005)
    ax.yaxis.set_major_locator(y_major_locator)

    # ax.yaxis.label.set_size(fontsize)  # 不管用

    ax.set_xlabel('The size of the image patches', fontsize=fontsize)
    ax.set_ylabel('performance', fontsize=fontsize)
    plt.xlim(-0.1, 4.4)
    if mode == 'ebar_with_cap' or mode == 'ebar_without_cap':
        # plt.ylim(mean_SRCC[-1] - error_SRCC[-1] - 5e-4)
        pass
    plt.legend(fontsize=fontsize)

    plt.tight_layout()
    figname = Path(__file__).name[:-3] + dataName+ mode + '.png'
    plt.savefig(figname, dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    mode1 = 'ebar_without_cap'
    mode2 = 'ebar_with_cap'
    mode3 = 'just plot'


    # plot_with_ebar(SRCC_Mean, SRCC_Error,PLCC_Mean, PLCC_Error, mode1)
    plot_with_ebar(SRCC_Mean, SRCC_Error,PLCC_Mean, PLCC_Error, mode2)
    # plot_with_ebar(SRCC_Mean, SRCC_Error,PLCC_Mean, PLCC_Error, mode3)
