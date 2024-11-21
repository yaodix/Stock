
# python 3.8+
import akshare as ak
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

# 添加数值、百分比
  
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
  close_price = np.asarray(data)
  diff = np.diff(close_price)
  diff = diff / close_price[:-1]
  diff = np.insert(diff, 0, 0)

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
        # 往回找到amp起点
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
    # 找下一个amp
    if first_trend_finded and to_find_up_trend:
      sum_res_p[idx] = max(diff[idx], sum_res_p[idx-1]+diff[idx])
      max_range = max(max_range, sum_res_p[idx])
      
      if max_range >= raise_thresh:
        # 往回找到amp起点
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
     
  if last_trend == 1: # 最后一波是amp，
    start_key = sorted(pivots.keys())[-1]
    last_idx = np.argmax(data[start_key:])
    pivots[start_key+int(last_idx)] = 1
  else:
    start_key = sorted(pivots.keys())[-1]
    last_idx = np.argmin(data[start_key:])
    pivots[start_key+int(last_idx)] = -1
  
    
  return pivots


def get_max_decade_ration(data):
  pass

def get_min_raise_ration(data):
  pass


def get_daily_raise_limit(close_data, code, thresh = 0.096):
  '''
  get daily raise limit idx
  ret:
    daily_limit_idx: amp的正向索引
    daily_limit_idx_revserse: amp的反向索引, 相对于最后一天的偏移
    diff: amp, first is zero
  '''
  raise_limit = thresh
  if ("30" in code[0:2]):
    raise_limit = thresh * 2

  close_price = np.asarray(close_data)
  diff = np.diff(close_price)
  diff_ratio = diff / close_price[:-1]
  diff_ratio = np.insert(diff_ratio, 0, 0)

  daily_limit_idx = []
  
  for idx, val in enumerate(diff_ratio):
    if val > raise_limit:
      daily_limit_idx.append(idx)

  daily_limit_idx = np.asarray(daily_limit_idx)
  daily_limit_idx_revserse = daily_limit_idx- close_data.__len__()
  return daily_limit_idx, daily_limit_idx_revserse, diff_ratio


# 添加测试用例
test_data_1 = np.array([1, 1.2, 1, 0.8, 1.5, 1.8, 1.74])
test_data_2 = np.array([1, 0.8, 1.2, 1, 0.5, 1.5, 1.8, 1.0, 1.03])
  
if __name__ == "__main__":
  # 下载个股日k数据图
  # df_daily = ak.stock_zh_a_hist(symbol="002952", period = "daily", start_date= "20230101", end_date="20240531")
  df_daily = ak.stock_zh_a_hist(symbol="002182", period = "daily", start_date= "20230102", end_date="20241215")
  # print(df_daily.tail())
  
  X = df_daily["Close"]

  data = np.asarray(X)    
  # data = test_data_1
  pivots = get_pivots(data, 0.06, 0.06)
  print(pivots)
  print(data[list(pivots.keys())])
  fig = plt.figure()
  plot_pivots(data, pivots)
  plot_pivot_line(data, pivots)
  
  plt.show()



