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

def plot_with_ebar_mean_std(mean_srcc, std_proposed, mean_plcc, std_baseline, data_name, category, save_dir,
                            mode='ebar_with_cap', font_size=13, horizontal_bias=0.05, vertical_bias=(2.7e-3, 0, -2.7e-3)):
    plt.figure()
    plt.grid(axis='y', ls='--')
    ax = plt.gca()
    ax.set_axisbelow(True)

    # sns.set_style(style='darkgrid')

    # x_ticks = [0, 1, 2, 3]
    # x_ticks = [0, 1, 2, 3, 4]

    CapSize = 3
    sp_average = (np.array(mean_plcc) + np.array(mean_srcc)) / 2

    plt.errorbar(x=range(5), y=mean_srcc, yerr=std_proposed, fmt='v--', label='SRCC', capsize=CapSize, linewidth=1)
    plt.errorbar(x=range(5), y=mean_plcc, yerr=std_baseline, fmt='^--', label='PLCC', capsize=CapSize, linewidth=1)
    plt.plot(sp_average, marker='d', label='average')


    for x, y in zip(range(5), mean_srcc):
        plt.annotate("%.3f" % y, xy=(x+horizontal_bias, y+vertical_bias[0]), bbox=dict(boxstyle='round,pad=0', fc='lavender', ec='k', lw=0, alpha=0.65), fontsize=font_size-1)
    for x, y in zip(range(5), sp_average):
        plt.annotate("%.3f" % y, xy=(x+horizontal_bias, y+vertical_bias[1]), bbox=dict(boxstyle='round,pad=0.05', fc=seaborn.xkcd_rgb['sandy yellow'], ec='k', lw=0.05, alpha=0.85), fontsize=font_size)
    for x, y in zip(range(5), mean_plcc):
        plt.annotate("%.3f" % y, xy=(x+horizontal_bias, y+vertical_bias[2]), bbox=dict(boxstyle='round,pad=0', fc='honeydew', ec='k', lw=0, alpha=0.65), fontsize=font_size-1)

    # for i in range(len(sp_average)):
    #     plt.text(i, sp_average[i] + vertical_bias, '%.3f' % sp_average[i], fontsize=font_size, backgroundcolor='silver', alpha=0.9)

    x_ticks = [0, 1, 2, 3, 4]
    x_ticks = ax.set_xticks(x_ticks)
    # xlabels = ax.set_xticklabels(['512', '256', '128', '64', '32'], rotation=0, fontsize=font_size)
    xlabels = ax.set_xticklabels(['32', '64', '128', '256', '512'], rotation=0, fontsize=font_size)


    # y_major_locator = MultipleLocator(0.005)
    # ax.yaxis.set_major_locator(y_major_locator)

    # ax.yaxis.label.set_size(fontsize)  # 不管用

    ax.set_xlabel('Patch Size', fontsize=font_size+2)
    ax.set_ylabel(category, fontsize=font_size+2)
    plt.xlim(-0.1, 4.45)
    if mode == 'ebar_with_cap' or mode == 'ebar_without_cap':
        # plt.ylim(mean_plcc[-1] - std_baseline[-1] - 5e-4, mean_srcc[0] + std_proposed[0] + 5e-4)
        pass
    plt.legend(fontsize=font_size, loc='best')

    plt.tight_layout()
    figname = Path(__file__).name[:-3] + data_name + category + mode + time.strftime('%m%d_%H%M%S') +'.png'
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





