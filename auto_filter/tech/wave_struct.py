'''
weekly wave structure support weekly
'''
import datetime as dt
import pandas as pd
import numpy as np
from  tqdm import tqdm
import os
import sys
import pickle
import mplfinance as mpf
import matplotlib.pyplot as plt
import cv2

cur_file = os.path.abspath(__file__)
start_index = cur_file.find("auto_filter")
pro_path = cur_file[:start_index+11]
sys.path.append(pro_path)

import tech.tech_base as tech_base
import data_utils

def shiftWaveStructure(df_one, pivots):
  '''
  修复波形的起点和终点
  '''

def waveStructureImplPivot(df_one):
  '''
  weekly horizon support
  '''
  price_1 = np.array(df_one["Close"])[-110:]
  pivots = tech_base.get_pivots(price_1, 0.15, 0.11)
  # data_utils.plot_pivots(price_1, pivots)
  # data_utils.plot_pivot_line(price_1, pivots)
  # plt.show()
  # print(price_1[pivots.keys()])
  
  # filter
  low_pivots_index = [k for k, v in pivots.items() if v == -1]
  high_pivots_index = [k for k, v in pivots.items() if v == 1]
  if (len(low_pivots_index) < 3):
    return False
  if (len(high_pivots_index) < 2):
    return False

  # 波谷在波峰前面，否则退出
  if (high_pivots_index[-1] > low_pivots_index[-1]):
    return False
    # 最新实时价格与最新低点价格对比
  last_price = np.asarray(price_1)[-1]
  last_lowest_pivot_price = np.asarray(price_1)[low_pivots_index[-1]]
  if last_price > 1.1 * last_lowest_pivot_price :
    return False

  raise_last_start_low_idx = low_pivots_index[-2]
  raise_last_end_high_idx = high_pivots_index[-1]
  raise_lastsecond_start_low_idx = low_pivots_index[-3]
  raise_lastsecond_end_high_idx = high_pivots_index[-2]
  # 下降数量大于2
  decade_last_end_low_idx = low_pivots_index[-1]

  if raise_last_start_low_idx - raise_lastsecond_end_high_idx < 3:
    return False
  
  if decade_last_end_low_idx - raise_last_end_high_idx < 3:
    return False

  ## 低点不能差多
  lastsecond_lowest_pivot_price = price_1[low_pivots_index[-2]]
  lastthird_lowest_pivot_price = price_1[low_pivots_index[-3]]
  if abs(last_lowest_pivot_price - lastsecond_lowest_pivot_price)/ lastsecond_lowest_pivot_price > 0.08 or \
     abs(lastsecond_lowest_pivot_price - lastthird_lowest_pivot_price)/ lastthird_lowest_pivot_price > 0.08  :
    return False
  
  # 高点不能差太多  
  last_high_pivot_price = price_1[high_pivots_index[-1]]
  lastsecond_high_pivot_price = price_1[high_pivots_index[-2]]
  if abs(last_high_pivot_price - lastsecond_high_pivot_price) / lastsecond_high_pivot_price > 0.5:
    return False
  
   # 两边时间间隔
  if 1 < abs(low_pivots_index[-1] - high_pivots_index[-1]) and  abs(low_pivots_index[-1] - high_pivots_index[-1]) < 16 and \
     1 < abs(low_pivots_index[-2] - high_pivots_index[-1]) and  abs(low_pivots_index[-2] - high_pivots_index[-1]) < 16 and \
     1 < abs(low_pivots_index[-2] - high_pivots_index[-2]) and  abs(low_pivots_index[-2] - high_pivots_index[-2]) < 16 and \
     1 < abs(low_pivots_index[-3] - high_pivots_index[-2]) and  abs(low_pivots_index[-3] - high_pivots_index[-2]) < 16:

    return True
    
  return False
  
  
def waveStructureImplApprox(df_one):
  price_1 = np.array(df_one["Close"])[-100:]
  price_list = []
  for i in range(1, len(price_1)):
    price_list.append([i, price_1[i]])
  price_list = np.array(price_list, dtype=np.float32)

  approx = cv2.approxPolyDP(price_list, 0.6, True)
  approx = approx.reshape((-1, 2))
  plt.plot(price_1)
  # plt.plot(approx, 'r--')
  # plt.scatter(approx[:, 0], approx[:, 1], c='r', marker='o')
  # plt.show()
  pass
def GetWaveStructureWeekly(df_dict):
  res_list = []
  for code, value in df_dict.items():
    sel = waveStructureImplPivot(value)
    if sel:
      res_list.append(code)
  return res_list


test_map = {
  "002891": ["20231027"],  #zhong chong gufen

}

def test():
  pass


if __name__ == '__main__':
  df_dict = data_utils.LoadPickleData(pro_path+"/sec_data/weekly.pickle")
  test_cnt = 0
  for key, val in test_map.items():
    test_cnt += val.__len__()

  test_dict = {}
  for key, val in test_map.items():
    test_dict[key] = df_dict[key]
    for date in val:
      end_day = dt.datetime.date(dt.datetime.strptime(date, "%Y%m%d"))
      test_dict[key] = test_dict[key][test_dict[key]["Date"] <= end_day]
    
  # test_dict = df_dict
  res_dict = GetWaveStructureWeekly(test_dict)
  print(res_dict)
