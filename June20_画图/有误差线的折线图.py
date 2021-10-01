"""-----------------------------------------------------
创建时间 :  2020/6/30  22:12
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


# 我的数据
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import seaborn as sns


# wc_t1_s = [0.913, 0.908, 0.906, 0.913, 0.910, 0.909, 0.911]
# wc_t1_p = [0.894, 0.887, 0.879, 0.888, 0.887, 0.882, 0.880]
#
# ck_t1_s = [0.912, 0.910]
# ck_t1_p = [0.884, 0.880]
#
# pc_t1_s = [0.908]  # , 0.910
# pc_t1_p = [0.886]  # , 0.889
#
# t1_s = wc_t1_s + ck_t1_s + pc_t1_s
# mt1_s = np.array(t1_s).mean()
# t1_p = wc_t1_p + ck_t1_p + pc_t1_p
# mt1_p = np.array(t1_p).mean()
#
# wc_n1_s = [0.894, 0.896, 0.892]
# wc_n1_p = [0.867, 0.869, 0.867]
#
# ck_n1_s = [0.892, 0.898, 0.895]
# ck_n1_p = [0.867, 0.877, 0.871]
#
# pc_n1_s = [0.894, 0.892, 0.896, 0.898]  # 这里大于0.9的是不能要的，要不然跟多次平均值那个图标可能会矛盾
# pc_n1_p = [0.867, 0.869, 0.871, 0.870]
#
# n1_s = wc_n1_s + ck_n1_s + pc_n1_s
# n1_s = np.array(n1_s)
# mn1_s = n1_s.mean()
#
#
# n1_p = wc_n1_p + ck_n1_p + pc_n1_p
# mn1_p = np.array(n1_p).mean()
#
# wc_t2_s = [0.906, 0.899]
# wc_t2_p = [0.888, 0.878]
#
# ck_t2_s = [0.909, 0.900, 0.909, 0.904, 0.900]
# ck_t2_p = [0.876, 0.873, 0.889, 0.882, 0.881]
#
# pc_t2_s = [0.908, 0.903, 0.904]  # 这俩比较小的可以替换掉
# pc_t2_p = [0.883, 0.880, 0.885]
#
# t2_s = wc_t2_s + ck_t2_s + pc_t2_s
# mt2_s = np.array(t2_s).mean()
#
# t2_p = wc_t2_p + ck_t2_p + pc_t2_p
# mt2_p = np.array(t2_p).mean()
#
# wc_n2_s = [0.882, 0.890, 0.880, 0.887]
# wc_n2_p = [0.861, 0.864, 0.862, 0.862]
#
# ck_n2_s = [0.889, 0.884, 0.891, 0.886]
# ck_n2_p = [0.871, 0.863, 0.867, 0.858]
#
# pc_n2_s = [0.893, 0.889]
# pc_n2_p = [0.861, 0.868]
#
# n2_s = wc_n2_s + ck_n2_s + pc_n2_s
# mn2_s = np.array(n2_s).mean()
#
# n2_p = wc_n2_p + ck_n2_p + pc_n2_p
# mn2_p = np.array(n2_p).mean()
#
# wc_t3_s = [0.903, 0.901, 0.897, 0.898, 0.901]
# wc_t3_p = [0.875, 0.878, 0.880, 0.885, 0.881]
#
# ck_t3_s = [0.897, 0.899, 0.899]
# ck_t3_p = [0.875, 0.874, 0.881]
#
# pc_t3_s = [0.904, 0.900]
# pc_t3_p = [0.889, 0.878]
#
# t3_s = wc_t3_s + ck_t3_s + pc_t3_s
# mt3_s = np.array(t3_s).mean()
#
# t3_p = wc_t3_p + ck_t3_p + pc_t3_p
# mt3_p = np.array(t3_p).mean()
#
# wc_n3_s = [0.885, 0.878, 0.890, 0.885]  # 好像是这个0.891不知道哪来的
# wc_n3_p = [0.864, 0.842, 0.867, 0.860]  # todo: 缺一个
#
# ck_n3_s = [0.876, 0.880, 0.881, 0.887]
# ck_n3_p = [0.859, 0.857, 0.859, 0.861]
#
# pc_n3_s = [0.880, 0.891]
# pc_n3_p = [0.849, 0.866]
#
# n3_s = wc_n3_s + ck_n3_s + pc_n3_s
# mn3_s = np.array(n3_s).mean()
#
# n3_p = wc_n3_p + ck_n3_p + pc_n3_p
# mn3_p = np.array(n3_p).mean()
#
# wc_t4_s = [0.895, 0.893, 0.904, 0.902, 0.896, 0.892, 0.896, 0.904]  # , 0.895, 0.904
# wc_t4_p = [0.874, 0.871, 0.887, 0.880, 0.863, 0.870, 0.874, 0.885]  # , 0.868, 0.877
#
# ck_t4_s = [0.898]
# ck_t4_p = [0.878]
#
# pc_t4_s = [0.901]
# pc_t4_p = [0.882]
#
# t4_s = wc_t4_s + ck_t4_s + pc_t4_s
# mt4_s = np.array(t4_s).mean()
#
# t4_p = wc_t4_p + ck_t4_p + pc_t4_p
# mt4_p = np.array(t4_p).mean()
#
# wc_n4_s = [0.886, 0.882, 0.887, 0.878, 0.869]
# wc_n4_p = [0.859, 0.851, 0.862, 0.846, 0.842]
#
# ck_n4_s = [0.874, 0.876, 0.887]
# ck_n4_p = [0.848, 0.858, 0.867]
#
# pc_n4_s = [0.873, 0.882]
# pc_n4_p = [0.853, 0.865]
#
# n4_s = wc_n4_s + ck_n4_s + pc_n4_s
# mn4_s = np.array(n4_s).mean()
#
# n4_p = wc_n4_p + ck_n4_p + pc_n4_p
# mn4_p = np.array(n4_p).mean()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

FONTSIZE = 14
mode = 'plcc'
if mode == 'srcc':
    excelPath = 'C:\\Users\\ZhangZiyan\\Desktop\\我的SRCC.xlsx'
    outPicPath = 'SRCC_withebar.jpg'
elif mode == 'plcc':
    excelPath = 'C:\\Users\\ZhangZiyan\\Desktop\\我的PLCC.xlsx'
    outPicPath = 'PLCC_withebar.jpg'

sns.set(style='darkgrid')
#读取数据
data = pd.read_excel(excelPath,sheet_name = 0)
std_table = data.groupby(by='Method').std() # 计算标准差
figdata = data.groupby(by='Method').mean() #计算均值
# 绘图
# 标记样式常用的值有（./,/o/v/^/s/*/D/d/x/</>/h/H/1/2/3/4/_/|）https://www.jianshu.com/p/b992c1279c73，参考
plt.errorbar(figdata.columns, figdata.loc['Proposed'],yerr=std_table.loc['Proposed'], fmt='d-', color='blue', ecolor='blue', label='proposed', capsize=3)
plt.errorbar(figdata.columns, figdata.loc['Baseline'],yerr=std_table.loc['Baseline'], fmt='s-', color='gray', ecolor='gray', label='baseline', capsize=3)

ax = sns.stripplot(x=figdata.columns, y=figdata.loc['Proposed'],data=data)

plt.xlabel('Number of Training Samples', fontsize=FONTSIZE)
plt.ylabel('SRCC', fontsize=FONTSIZE)

# plt.legend(fontsize=19,ncol=2)
# plt.tight_layout() #防止保存时大小不合适
plt.savefig(outPicPath, dpi=600, bbox_inches='tight')
plt.show()
