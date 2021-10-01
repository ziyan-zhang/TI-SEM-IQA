"""-----------------------------------------------------
创建时间 :  2020/7/3  20:54
说明    :
todo   :  所有的都是不加载权重
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


C512_s = [0.876]
C512_p = [0.839]

mC512_s = np.array(C512_s).mean()
C512_s_std = np.array(C512_s).std()

mC512_p = np.array(C512_p).mean()
C512_p_std = np.array(C512_p).std()


C256_s = [0.898, 0.895]
C256_p = [0.881, 0.875]

mC256_s = np.array(C256_s).mean()
C256_s_std = np.array(C256_s).std()

mC256_p = np.array(C256_p).mean()
C256_p_std = np.array(C256_p).std()


C128_s = [0.899, 0.900, 0.900]#0.904
C128_p = [0.871, 0.872, 0.872]#0.885

mC128_s = np.array(C128_s).mean()
C128_s_std = np.array(C128_s).std()

mC128_p = np.array(C128_p).mean()
C128_p_std = np.array(C128_p).std()


C64_s = [0.880, 0.886, 0.897]
C64_p = [0.855, 0.858, 0.876]

mC64_s = np.array(C64_s).mean()
C64_s_std = np.array(C64_s).std()

mC64_p = np.array(C64_p).mean()
C64_p_std = np.array(C64_p).std()


C32_s = [0.858, 0.851, 0.853]
C32_p = [0.837, 0.822, 0.822]

mC32_s = np.array(C32_s).mean()
C32_s_std = np.array(C32_s).std()

mC32_p = np.array(C32_p).mean()
C32_p_std = np.array(C32_p).std()


SRCC_Mean = [mC512_s, mC256_s, mC128_s, mC64_s, mC32_s]
SRCC_Error = [C512_s_std, C256_s_std, C128_s_std, C64_s_std, C32_s_std]
SRCC_Len = [len(C512_s), len(C256_s), len(C128_s), len(C64_s), len(C32_s)]

PLCC_Mean = [mC512_p, mC256_p, mC128_p, mC64_p, mC32_p]
PLCC_Error = [C512_p_std, C256_p_std, C128_p_std, C64_p_std, C32_p_std]
PLCC_Len = [len(C512_p), len(C256_p), len(C128_p), len(C64_p), len(C32_p)]

dataName = Path(__file__).name[:-3]

if __name__ == '__main__':
    fig, ax = plt.subplots()
    sns.set(style='darkgrid')
    plt.plot(C512_s, label='C512', color='red', linestyle='-')
    plt.plot(C256_s, label='C256', color='green', linestyle='-')
    plt.plot(C128_s, label='C128', color='blue', linestyle='-')
    plt.plot(C64_s, label='C64', color='cornflowerblue', linestyle='-')
    plt.plot(C32_s, label='C32', color='cornflowerblue', linestyle='-')

    plt.legend()
    plt.show()