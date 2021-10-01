"""-----------------------------------------------------
创建时间 :  2020/6/30  21:01
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import load_diabetes

def fun(x):
    if x >0:
        return 1
    else:
        return 0
# sklearn自带数据 diabetes 糖尿病数据集
diabetes=load_diabetes()
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
#只抽取前80个数据
df=data[:80]
#由于diabetes中的数据均已归一化处理过，sex列中的值也归一化，现将其划分一下，大于0的设置为1，小于等于0的设置为0
df['sex']=df['sex'].apply(lambda x: fun(x))

"""
案例1：绘制带有误差带的单线图，显示置信区间
"""
ax = sns.lineplot(x="age", y="s4",data=df)
plt.show()

