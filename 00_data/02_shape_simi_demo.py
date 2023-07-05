# 20230704
# python 3.8+

import akshare as ak
import numpy as np
from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd
from shapesimilarity import shape_similarity
import similaritymeasures
import math
# import zigzag 

# 下载A股所有数据
# all_df = ak.stock_zh_a_spot_em()
# print(all_df.head())

# 20230131-20230531,　形态, 83周期
# target: 600640，20230303 - 20030705

# 下载个股日k数据图
df_daily = ak.stock_zh_a_hist(symbol="000338", period = "daily", start_date= "20230131", end_date="20230531")
t_df_daily = ak.stock_zh_a_hist(symbol="600640", period = "daily", start_date= "20230302", end_date="20230704")


X = df_daily["收盘"]
x_t = t_df_daily["收盘"]
# pivots = zigzag.peak_valley_pivots(X, 0.03, -0.03)
# t_pivots = zigzag.peak_valley_pivots(x_t, 0.03, -0.03)

# 添加数值、百分比
def plot_pivots(X, pivots):
    plt.xlim(0, len(X))
    plt.ylim(X.min()*0.99, X.max()*1.01)
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    plt.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='g')
    plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')
      

# plot_pivots(X, pivots)
# plt.show()

x = np.linspace(1, -1, num=83)

shape1 = np.column_stack((x, X))
shape2 = np.column_stack((x, x_t))

similarity = shape_similarity(shape1, shape2)
# Discrete Frechet distance
print(f"shape_similarity similarity {similarity}")




