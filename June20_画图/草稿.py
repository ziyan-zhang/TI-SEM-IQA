import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 6)
y = x * x

plt.plot(x, y, marker='o')
for xy in zip(x, y):  # 这里xy不是相乘，是一个坐标点。
    plt.annotate("(%s,%s)" % xy, xy=xy, xytext=(-20, 10), textcoords='offset points',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.5))
plt.show()

