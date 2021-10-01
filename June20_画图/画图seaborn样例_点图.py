"""-----------------------------------------------------
创建时间 :  2020/7/1  10:56
说明    :
todo   :  
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import seaborn as sns

tips = sns.load_dataset("tips")
ax = sns.stripplot(x="day", y="total_bill", data=tips)