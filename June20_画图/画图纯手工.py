"""-----------------------------------------------------
创建时间 :  2020/7/1  14:45
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import time
# from July2nd_overlap_新数据 import Proposed_Plcc_Mean, Baseline_Plcc_Mean, Proposed_Plcc_Error, Baseline_Plcc_Error,\
#     Proposed_Srcc_Mean, Baseline_Srcc_Mean, Proposed_Srcc_Error, Baseline_Srcc_Error, Proposed_Srcc_Len, Baseline_Srcc_Len
# from 清洗后的加上后来未清洗的数据 import Proposed_Plcc_Mean, Baseline_Plcc_Mean, Proposed_Plcc_Error, Baseline_Plcc_Error,\
#     Proposed_Srcc_Mean, Baseline_Srcc_Mean, Proposed_Srcc_Error, Baseline_Srcc_Error, Proposed_Srcc_Len, Baseline_Srcc_Len
from 数据文昌新 import Proposed_Plcc_Mean, Baseline_Plcc_Mean, Proposed_Plcc_Error, Baseline_Plcc_Error,\
    Proposed_Srcc_Mean, Baseline_Srcc_Mean, Proposed_Srcc_Error, Baseline_Srcc_Error, Proposed_Srcc_Len, Baseline_Srcc_Len, dataName
from pathlib import Path
from matplotlib.pyplot import MultipleLocator
import numpy as np


def plot_with_ebar(mean_proposed, error_proposed, mean_baseline, error_baseline, category, mode='ebar_with_cap',
                   fontsize=13, vertical_bias=6e-4):
    sns.set_style(style='darkgrid')
    fig, ax = plt.subplots()
    # x_ticks = [0, 1, 2, 3]
    x_ticks = [0, 1, 2, 3, 4]

    if mode == 'ebar_without_cap':
        CapSize = 0
        plt.errorbar(x=x_ticks, y=mean_proposed, yerr=error_proposed, fmt='d-', color='blue', ecolor='blue',
                     label='proposed', capsize=CapSize)
        plt.errorbar(x=x_ticks, y=mean_baseline, yerr=error_baseline, fmt='s-', color='gray', ecolor='gray',
                     label='baseline', capsize=CapSize)
    elif mode == 'ebar_with_cap':
        CapSize = 3
        plt.errorbar(x=x_ticks, y=mean_proposed, yerr=error_proposed, fmt='d-', color='blue', ecolor='blue',
                     label='proposed', capsize=CapSize)
        plt.errorbar(x=x_ticks, y=mean_baseline, yerr=error_baseline, fmt='s-', color='gray', ecolor='gray',
                     label='baseline', capsize=CapSize)
    elif mode == 'just plot':
        plt.plot(mean_proposed, marker='d', color='blue', label='proposed')
        plt.plot(mean_baseline, marker='s', color='gray', label='baseline')
    else:
        raise ValueError

    for i in range(len(mean_proposed)):
        plt.text(i, mean_proposed[i] + vertical_bias, '%.4f_' % mean_proposed[i]+str(Proposed_Srcc_Len[i]), fontsize=fontsize)
        plt.text(i, mean_baseline[i] + vertical_bias, '%.4f_' % mean_baseline[i]+str(Baseline_Srcc_Len[i]), fontsize=fontsize)

    x_ticks = ax.set_xticks(x_ticks)
    xlabels = ax.set_xticklabels(['100%', '80%', '60%', '40%', '20%'], rotation=0, fontsize=fontsize)

    y_major_locator = MultipleLocator(0.005)
    ax.yaxis.set_major_locator(y_major_locator)

    # ax.yaxis.label.set_size(fontsize)  # 不管用

    ax.set_xlabel('number of training samples', fontsize=fontsize)
    ax.set_ylabel(category, fontsize=fontsize)
    plt.xlim(-0.1, 4.4)
    if mode == 'ebar_with_cap' or mode == 'ebar_without_cap':
        # plt.ylim(mean_baseline[-1] - error_baseline[-1] - 5e-4, mean_proposed[0] + error_proposed[0] + 5e-4)
        pass
    plt.legend(fontsize=fontsize)

    plt.tight_layout()
    figname = Path(__file__).name[:-3] + dataName + category + mode + time.strftime('%m%d_%H%M%S') +'.png'
    plt.savefig(figname, dpi=600, bbox_inches='tight')
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



