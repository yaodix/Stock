
# 20230704
# python 3.8+
from scipy import interpolate
import akshare as ak
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

# 添加数值、百分比
def plot_pivots(X, pivots):
    plt.xlim(0, len(X))
    plt.ylim(X.min()*0.99, X.max()*1.01)
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    high_idx =[]
    low_idx =[]
    for key in  pivots.keys():
      if pivots[key] == 1:
        high_idx.append(key)
      else:
        low_idx.append(key)
    sorted(low_idx)
    sorted(high_idx)
      
    plt.scatter(high_idx, X[high_idx], color='r')
    plt.scatter(low_idx, X[low_idx], color='g')
    
    
def plot_pivot_line(X, pivots, enable_support = True, enable_resistance = True):
      ### fit low pivots
    # keep only -1 values
    if enable_support:
      low_pivots_index = [k for k, v in pivots.items() if v == -1]
      y = X[low_pivots_index]
      x = low_pivots_index

      data_cnt = len(X)
      data_range = range(0, data_cnt) 
      akima_interpolator = interpolate.Akima1DInterpolator(x, y)
      x_fit = np.linspace(min(data_range), max(data_range), data_cnt*2)
      y_fit = akima_interpolator(x_fit)
      plt.plot(x_fit, y_fit,'b')
    if enable_resistance:
      low_pivots_index = [k for k, v in pivots.items() if v == 1]
      y = X[low_pivots_index]
      x = low_pivots_index

      data_cnt = len(X)
      data_range = range(0, data_cnt) 
      akima_interpolator = interpolate.Akima1DInterpolator(x, y)
      x_fit = np.linspace(min(data_range), max(data_range), data_cnt*2)
      y_fit = akima_interpolator(x_fit)
      plt.plot(x_fit, y_fit,'r')

  
  
def get_pivots(data, raise_thresh = 0.1, fall_thresh = 0.07):
  '''
    data: Close array
    raise_thresh: raise ratio threshold
  '''
  pivots = {}  # -1:low,1:high
  last_trend = 0 # -1: low, 1: high, 记录最后一次的趋势
  first_trend_finded = False
  to_find_up_trend = False
  to_find_down_trend = False
  range_data = np.asarray(data)
  diff = np.diff(range_data)
  diff = np.insert(diff, 0, 0)
  diff = diff / range_data
  sum_res_n = np.zeros_like(diff) # 用于找最大负值
  sum_res_p = np.zeros_like(diff) # 用于找最大正值
  max_range = 0  
  min_range = 0
  
  back_sum = 0
  for idx, num in enumerate(diff):
    if idx == 0: 
      continue  # 跳过0
    # print(idx)
    if first_trend_finded is False:
      sum_res_p[idx] = max(diff[idx], sum_res_p[idx-1]+diff[idx])
      sum_res_n[idx] = min(diff[idx], sum_res_n[idx-1]+diff[idx])
      max_range = max(max_range, sum_res_p[idx])
      min_range = min(min_range, sum_res_n[idx])
      # print(idx)
      if max_range >= raise_thresh:
        # 往回找到涨幅起点
        for back_idx in range(idx, -1, -1):
          back_sum = back_sum + diff[back_idx]
          if abs(back_sum - max_range) < 1e-6:
            pivots[back_idx-1] = -1
            first_trend_finded = True
            to_find_down_trend = True
            max_range = 0
            min_range = 0
            last_trend = 1
            break
      if min_range <= -fall_thresh:
        # 往回找到跌幅起点
        back_sum = 0        
        for back_idx in range(idx, -1, -1):
          back_sum = back_sum + diff[back_idx]
          if abs(back_sum - min_range) < 1e-6:
            pivots[back_idx-1] = 1
            first_trend_finded = True
            to_find_up_trend = True
            min_range =0
            max_range =0
            last_trend = -1
            break
    # 找下一个涨幅
    if first_trend_finded and to_find_up_trend:
      sum_res_p[idx] = max(diff[idx], sum_res_p[idx-1]+diff[idx])
      max_range = max(max_range, sum_res_p[idx])
      
      if max_range >= raise_thresh:
        # 往回找到涨幅起点
        back_sum = 0        
        for back_idx in range(idx, -1, -1):
          back_sum = back_sum+ diff[back_idx]
          if abs(back_sum - max_range) < 1e-6:
            pivots[back_idx-1] = -1
            to_find_up_trend = False
            to_find_down_trend = True
            min_range = 0
            max_range = 0
            last_trend = 1
            break
    # 找下一个跌幅    
    if first_trend_finded and to_find_down_trend:
      sum_res_n[idx] = min(diff[idx], sum_res_n[idx-1]+diff[idx])
      min_range = min(min_range, sum_res_n[idx])

      if min_range <= -fall_thresh:
        # 往回找到跌幅起点
        back_sum = 0        
        for back_idx in range(idx, -1, -1):
          back_sum += diff[back_idx]
          if abs(back_sum - min_range) < 1e-6:
            pivots[back_idx-1] = 1
            to_find_down_trend = False
            to_find_up_trend = True
            min_range = 0
            max_range = 0
            last_trend = -1            
            break  
  if len(pivots) < 1:
    return pivots
     
  if last_trend == 1: # 最后一波是涨幅，
    start_key = sorted(pivots.keys())[-1]
    last_idx = np.argmax(data[start_key:])
    pivots[start_key+last_idx] = 1
  else:
    start_key = sorted(pivots.keys())[-1]
    last_idx = np.argmin(data[start_key:])
    pivots[start_key+last_idx] = -1
  
    
  return pivots


# 添加测试用例
test_data_1 = np.array([1, 1.2, 1, 0.8, 1.5, 1.8, 1.74])
test_data_2 = np.array([1, 0.8, 1.2, 1, 0.5, 1.5, 1.8, 1.0, 1.03])
  
if __name__ == "__main__":
  # 下载个股日k数据图
  # df_daily = ak.stock_zh_a_hist(symbol="002952", period = "daily", start_date= "20230101", end_date="20240531")
  df_daily = ak.stock_zh_a_hist(symbol="002182", period = "daily", start_date= "20230102", end_date="20241215")
  # print(df_daily.tail())
  
  X = df_daily["收盘"]

  data = np.asarray(X)    
  # data = test_data_1
  pivots = get_pivots(data, 0.06, 0.06)
  print(pivots)
  print(data[list(pivots.keys())])
  fig = plt.figure()
  plot_pivots(data, pivots)
  plot_pivot_line(data, pivots)
  
  plt.show()



