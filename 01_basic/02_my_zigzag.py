# 20230704
# python 3.8+

import akshare as ak
import numpy as np
import pandas as pd
from findiff import FinDiff

import matplotlib.pyplot as plt


# 下载A股所有数据
# all_df = ak.stock_zh_a_spot_em()
# print(all_df.head())

# 20230131-20230531,　形态, 83周期
# target: 600640，20230303 - 20030705

# 下载个股日k数据图
df_daily = ak.stock_zh_a_hist(symbol="000338", period = "daily", start_date= "20230131", end_date="20230531")
t_df_daily = ak.stock_zh_a_hist(symbol="600640", period = "daily", start_date= "20230302", end_date="20230704")
print(df_daily.tail())

data_val = df_daily[["日期", "收盘"]]
X = t_df_daily["收盘"]
print(X.tail())

data = np.asarray(X)

# 添加数值、百分比
def plot_pivots(X, pivots):
    plt.xlim(0, len(X))
    plt.ylim(X.min()*0.99, X.max()*1.01)
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    # plt.plot(np.arange(len(X)), pivots, 'r-', alpha=0.5)
    # plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    plt.scatter(pivots.keys(), X[pivots.keys()], color='g')
      


def moving_average(x, w):
    tmp = np.convolve(x, np.ones(w), 'same') / w
    half_w = int(w/2)
    tmp[:half_w] = x[:half_w]
    tmp[-half_w:] = x[-half_w:]
    return tmp
  

def my_zigzag(data, valid_thersh = 0.1):
  pivots = {}  # -1:low, 0: horizon, 1:high
  adj_diff = np.diff(data)
  base_price = data[0]
  # 一维前缀和
  cum_sum = np.cumsum(adj_diff)

  # 先找到一个10%变化的阶段，判断起步方向
  find_up = False
  find_down = False
  start_index = 0
  for index, val in enumerate(cum_sum):
    if (not find_down and not find_up):
      if abs(val / base_price) > valid_thersh:
        if val > cum_sum[0]: 
          pivots[0] = -1
          find_up = True
          start_index = index
        else:
          pivots[0] = 1
          find_down = True
          start_index = index
        
    if (find_up and not find_down):  # 查找下降至少超过10%连续子数组
      if index > start_index:
        find_val = False
        max_diff = 0
        max_index = 0
        for gap_index in range(start_index, index+1):
          if cum_sum[index] < cum_sum[gap_index]:
            diff = abs((cum_sum[index] - cum_sum[gap_index]) / data[index+1])
            if diff > valid_thersh:  # down
              # 遍历查找幅度最大的值
              if (diff > max_diff):
                max_diff = diff
                max_index = gap_index
                find_val = True          
            
        if find_val:
          pivots[max_index+1] = 1
          find_down = True
          find_up = False
          start_index = index
          
          
    if (find_down and not find_up):  # 查找上升
      if index > start_index:
        find_val = False
        max_diff = 0
        max_index = 0
        for gap_index in range(start_index, index+1):
          if cum_sum[index] > cum_sum[gap_index]:
            diff = abs((cum_sum[index] - cum_sum[gap_index]) / data[gap_index+1])
            if diff > valid_thersh:  # down
              # 遍历查找幅度最大的值
              if (diff > max_diff):
                max_diff = diff
                max_index = gap_index
                find_val = True          
            
        if find_val:
          pivots[max_index+1] = 1
          find_down = False
          find_up = True
          start_index = index
          
  pivots[int(data.__len__()-1)] = -list(pivots.values())[-1]
  
  return pivots
  
# 在某个涨跌趋势中，获取水平震荡期数据
# def get_hor_data(data, thresh_limit = 0.06):
  
# ma5 = moving_average(X, 5);

# reversed_arr = data[::-1]
# reverse to find pivots
pivots = my_zigzag(data)

          

plot_pivots(X, pivots)
plt.show()



