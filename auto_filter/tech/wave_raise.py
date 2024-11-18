import sys
import os
import akshare as ak
import numpy as np
import matplotlib.pyplot as plt
from  tqdm import tqdm
import datetime as dt

cur_file = os.path.abspath(__file__)
start_index = cur_file.find("auto_filter")
pro_path = cur_file[:start_index+11]
sys.path.append(pro_path)

import tech.tech_base as tech_base
import data_utils

def count_numbers_in_range(arr, start, end):
    '''
    count nonzero numbers in range
    '''
    # 使用条件索引来选择在指定区间内的数字
    mask = (arr >= start) & (arr <= end)
    # 统计满足条件的数字个数
    count = np.count_nonzero(mask)
    return count

def filter_low_wave(src_data, pivots, security_code, verbose = False):
  raise_limit_idx, idx_r, ratio =  tech_base.get_daily_raise_limit(src_data["Close"], security_code)
  raise_limit = 0.096 
  if ("30" in security_code[0:2]):
    raise_limit = 0.198

  low_pivots_index = [k for k, v in pivots.items() if v == -1]
  high_pivots_index = [k for k, v in pivots.items() if v == 1]
  if (len(low_pivots_index) < 3):
    return False
  if (len(high_pivots_index) < 2):
    return False
  
  # 波谷在波峰前面，否则退出
  if (high_pivots_index[-1] > low_pivots_index[-1]):
    return False
  
  # 涨停数量不少于1个，不多于2个
  raise_last_start_low_idx = low_pivots_index[-2]
  raise_last_end_high_idx = high_pivots_index[-1]
  count1 = count_numbers_in_range(raise_limit_idx, raise_last_start_low_idx, raise_last_end_high_idx)
  if count1 < 1 or count1 > 2:  
    return False

  raise_lastsecond_start_low_idx = low_pivots_index[-3]
  raise_lastsecond_end_high_idx = high_pivots_index[-2]
  count2 = count_numbers_in_range(raise_limit_idx, raise_lastsecond_start_low_idx, raise_lastsecond_end_high_idx)
  if count2 < 1 or count2 > 2:
    return False
  
  # 2个低点价格差距不能大
  if abs(src_data["Close"][low_pivots_index[-1]] - src_data["Close"][low_pivots_index[-2]]) / src_data["Close"][low_pivots_index[-2]] >  0.15:
    return False

  # 下降数量大于等于2
  decade_last_end_low_idx = low_pivots_index[-1]
  if raise_last_start_low_idx - raise_lastsecond_end_high_idx < 2:
    return False
  
  if decade_last_end_low_idx - raise_last_end_high_idx < 1:
    return False
  
    # 增加涨幅比例限制
  raise_1 = abs(src_data["Close"][high_pivots_index[-2]]-src_data["Close"][low_pivots_index[-3]])  /src_data["Close"][low_pivots_index[-3]]
  raise_2 = abs(src_data["Close"][high_pivots_index[-1]]-src_data["Close"][low_pivots_index[-2]]) / src_data["Close"][low_pivots_index[-2]]
  fail_1 = abs(src_data["Close"][high_pivots_index[-2]]-src_data["Close"][low_pivots_index[-2]])/src_data["Close"][low_pivots_index[-2]]
  fail_2 = abs(src_data["Close"][high_pivots_index[-1]]-src_data["Close"][low_pivots_index[-1]])/src_data["Close"][low_pivots_index[-1]]
  if  (max(raise_1, raise_2) - min(raise_2, raise_1)) > 0.3 :
    return False
  
  # 增加跌幅比例限制
  if  (max(fail_1, fail_2) - min(fail_2, fail_1)) > 0.3:
    return False

  # 涨幅
  raise_lastsecond = abs(src_data["Close"][raise_lastsecond_end_high_idx] - src_data["Close"][raise_lastsecond_start_low_idx]) / src_data["Close"][raise_lastsecond_start_low_idx]
  raise_last = abs(src_data["Close"][raise_last_end_high_idx] - src_data["Close"][raise_last_start_low_idx]) / src_data["Close"][raise_last_start_low_idx]
  if raise_lastsecond < 1.8 * raise_limit or raise_last < 1.8 * raise_limit:
    return True

  return False 
  

def filter_high_wave(src_data, pivots, security_code, verbose = False):
  raise_limit_idx, idx_r, ratio =  tech_base.get_daily_raise_limit(src_data["Close"], security_code)  
  raise_limit = 0.096
  if ("30" in security_code[0:2]):
    raise_limit = 0.198

  low_pivots_index = [k for k, v in pivots.items() if v == -1]
  high_pivots_index = [k for k, v in pivots.items() if v == 1]
  if (len(low_pivots_index) < 3):
    return False
  if (len(high_pivots_index) < 2):
    return False

  # 波谷在波峰前面，否则退出
  if (high_pivots_index[-1] > low_pivots_index[-1]):
    return False
  
  # 涨停数量不少于1个，不多于2个
  raise_last_start_low_idx = low_pivots_index[-2]
  raise_last_end_high_idx = high_pivots_index[-1]
  count1 = count_numbers_in_range(raise_limit_idx, raise_last_start_low_idx, raise_last_end_high_idx)
  if count1 < 1 or count1 > 4:  
    return False

  raise_lastsecond_start_low_idx = low_pivots_index[-3]
  raise_lastsecond_end_high_idx = high_pivots_index[-2]
  count2 = count_numbers_in_range(raise_limit_idx, raise_lastsecond_start_low_idx, raise_lastsecond_end_high_idx)
  if count2 < 1 or count2 > 4:
    return False
  
  # 最新实时价格与最新低点价格对比
  last_price = np.asarray(src_data["Close"])[-1]
  last_lowest_pivot_price = np.asarray(src_data["Close"])[low_pivots_index[-1]]
  if last_price > 1.05 * last_lowest_pivot_price :
    return False
  
  # 下降数量大于2
  decade_last_end_low_idx = low_pivots_index[-1]
  if raise_last_start_low_idx - raise_lastsecond_end_high_idx < 3:
    return False
  
  if decade_last_end_low_idx - raise_last_end_high_idx < 3:
    return False

  # 低点不能越来越低
  lastsecond_lowest_pivot_price = src_data["Close"][low_pivots_index[-2]]
  lastthird_lowest_pivot_price = src_data["Close"][low_pivots_index[-3]]
  
  last_high_pivot_price = src_data["Close"][high_pivots_index[-1]]
  lastsecond_high_pivot_price = src_data["Close"][high_pivots_index[-2]]

  # if not (last_high_pivot_price > lastsecond_high_pivot_price and \
  #    last_lowest_pivot_price > lastsecond_lowest_pivot_price and \
  #    lastsecond_lowest_pivot_price > lastthird_lowest_pivot_price):
  #   return False
  if last_lowest_pivot_price < lastsecond_lowest_pivot_price*0.95 or \
     lastsecond_lowest_pivot_price < lastthird_lowest_pivot_price*0.95 :
    return False
  
   # 两边时间间隔
  if 1 < abs(low_pivots_index[-1] - high_pivots_index[-1]) and  abs(low_pivots_index[-1] - high_pivots_index[-1]) < 15 and \
     1 < abs(low_pivots_index[-2] - high_pivots_index[-1]) and  abs(low_pivots_index[-2] - high_pivots_index[-1]) < 15 and \
     1 < abs(low_pivots_index[-2] - high_pivots_index[-2]) and  abs(low_pivots_index[-2] - high_pivots_index[-2]) < 15 and \
     1 < abs(low_pivots_index[-3] - high_pivots_index[-2]) and  abs(low_pivots_index[-3] - high_pivots_index[-2]) < 15:

    return True
    
  return False
  
def waveTechFilter(df_dict, enable_high = True, enable_low = True):
  '''
  ret: code_pivot dict
  '''
  wave_low_dict = {}
  wave_high_dict = {}
  for code, df_daily in tqdm(df_dict.items()):  
    close = df_daily["Close"]    
    close = np.asarray(close)    

    if enable_high:
      pivots_high_wave = tech_base.get_pivots(close, 0.15, 0.11)
      # data_utils.plot_pivots(df_daily["Close"], pivots_high_wave)
      # data_utils.plot_pivot_line(df_daily["Close"], pivots_high_wave)

      sel1 = filter_high_wave(df_daily, pivots_high_wave, code)
      if sel1:
        wave_high_dict[code] = pivots_high_wave
    if enable_low:
      pivots_low_wave = tech_base.get_pivots(close, 0.096, 0.05)
      sel2 = filter_low_wave(df_daily, pivots_low_wave, code)
      # data_utils.plot_pivots(df_daily["Close"], pivots_low_wave)
      # data_utils.plot_pivot_line(df_daily["Close"], pivots_low_wave)
      if sel2:
        wave_low_dict[code] = pivots_low_wave

  return wave_low_dict, wave_high_dict


def test_low_wave(df_dict, test_map):
  test_cnt = 0
  for key, val in test_map.items():
    test_cnt += val.__len__()

  test_dict = {}
  for key, val in test_map.items():
    test_dict[key] = df_dict[key]
    for date in val:
      end_day = dt.datetime.date(dt.datetime.strptime(date, "%Y%m%d"))
      test_dict[key] = test_dict[key][test_dict[key]["Date"] <= end_day]
    
      wave_low_dict, wave_high_dict = waveTechFilter(test_dict, enable_high = False)
  # print(wave_low_dict)
  return wave_low_dict

def test_high_wave(df_dict, test_map):
  test_cnt = 0
  for key, val in test_map.items():
    test_cnt += val.__len__()

  test_dict = {}
  for key, val in test_map.items():
    test_dict[key] = df_dict[key]
    for date in val:
      end_day = dt.datetime.date(dt.datetime.strptime(date, "%Y%m%d"))
      test_dict[key] = test_dict[key][test_dict[key]["Date"] <= end_day]
    
      wave_low_dict, wave_high_dict = waveTechFilter(test_dict, enable_low = False)
  # print(wave_high_dict)
  return wave_high_dict

'''
low wave测试集合
'''
high_wave_dict = {                
                 "000627": ["20241017"],
                 "603787": ["20241022"],
                 "000859": ["20241011"],
                 }

low_wave_dict = {
  "000826": ["20241104"],
  "603787": ["20240408", "20240428"],
  "603196": ["20240829"],   

}

if __name__ == "__main__":
  pickle_path = '/home/yao/workspace/Stock/auto_filter/sec_data/daily.pickle' 
  df_dict = data_utils.LoadPickleData(pickle_path)
  wave_low_dict = test_low_wave(df_dict, low_wave_dict)
  print(f"wave_low_dict {wave_low_dict.keys()}")
  # wave_high_dict = test_high_wave(df_dict, high_wave_dict)
  # print(f"wave_high_dict {wave_high_dict.keys()}")

