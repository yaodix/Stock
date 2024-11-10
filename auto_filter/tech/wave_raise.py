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
    # 使用条件索引来选择在指定区间内的数字
    mask = (arr >= start) & (arr <= end)
    # 统计满足条件的数字个数
    count = np.count_nonzero(mask)
    return count

def filter_low_wave(src_data, pivots, security_code, verbose = False):
  raise_limit_idx, idx_r, ratio =  tech_base.get_daily_raise_limit(src_data["收盘"], security_code)
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
  if abs(src_data["收盘"][low_pivots_index[-1]] - src_data["收盘"][low_pivots_index[-2]]) / src_data["收盘"][low_pivots_index[-2]] >  0.15:
    return False

  # 下降数量大于2
  decade_last_end_low_idx = low_pivots_index[-1]
  if raise_last_start_low_idx - raise_lastsecond_end_high_idx < 3:
    return False
  
  if decade_last_end_low_idx - raise_last_end_high_idx < 3:
    return False
  
    # 增加涨幅比例限制
  raise_1 = abs(src_data["收盘"][high_pivots_index[-2]]-src_data["收盘"][low_pivots_index[-3]])  /src_data["收盘"][low_pivots_index[-3]]
  raise_2 = abs(src_data["收盘"][high_pivots_index[-1]]-src_data["收盘"][low_pivots_index[-2]]) / src_data["收盘"][low_pivots_index[-2]]
  fail_1 = abs(src_data["收盘"][high_pivots_index[-2]]-src_data["收盘"][low_pivots_index[-2]])/src_data["收盘"][low_pivots_index[-2]]
  fail_2 = abs(src_data["收盘"][high_pivots_index[-1]]-src_data["收盘"][low_pivots_index[-1]])/src_data["收盘"][low_pivots_index[-1]]
  if max(raise_1, raise_2) / min(raise_2, raise_1)  > 2:
    return False
  
  # 增加跌幅比例限制
  if max(fail_1, fail_2) / min(fail_2, fail_1)  > 1.6:
    return False

  # 涨幅
  raise_lastsecond = abs(src_data["收盘"][raise_lastsecond_end_high_idx] - src_data["收盘"][raise_lastsecond_start_low_idx]) / src_data["收盘"][raise_lastsecond_start_low_idx]
  raise_last = abs(src_data["收盘"][raise_last_end_high_idx] - src_data["收盘"][raise_last_start_low_idx]) / src_data["收盘"][raise_last_start_low_idx]
  if raise_lastsecond < 1.8 * raise_limit or raise_last < 1.8 * raise_limit:
    return True

  return False 
  

def filter_high_wave(src_data, pivots, security_code, verbose = False):
  raise_limit_idx, idx_r, ratio =  tech_base.get_daily_raise_limit(src_data["收盘"], security_code)  
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
  if count1 < 1:  
    return False

  raise_lastsecond_start_low_idx = low_pivots_index[-3]
  raise_lastsecond_end_high_idx = high_pivots_index[-2]
  count2 = count_numbers_in_range(raise_limit_idx, raise_lastsecond_start_low_idx, raise_lastsecond_end_high_idx)
  if count2 < 1:
    return False
  
  # 收盘价格与前低点价格对比
  last_price = np.asarray(src_data["收盘"])[-1]
  last_lowest_pivot_price = np.asarray(src_data["收盘"])[low_pivots_index[-1]]
  if last_price > 1.05 * last_lowest_pivot_price :
    return False
  
  # 下降数量大于2
  decade_last_end_low_idx = low_pivots_index[-1]
  if raise_last_start_low_idx - raise_lastsecond_end_high_idx < 3:
    return False
  
  if decade_last_end_low_idx - raise_last_end_high_idx < 3:
    return False

  # 高点更高，低点也更高
  lasrsecond_lowest_pivot_price = src_data["收盘"][low_pivots_index[-2]]
  lastthird_lowest_pivot_price = src_data["收盘"][low_pivots_index[-3]]
  
  last_high_pivot_price = src_data["收盘"][high_pivots_index[-1]]
  lastsecond_high_pivot_price = src_data["收盘"][high_pivots_index[-2]]

  if not (last_high_pivot_price > lastsecond_high_pivot_price and \
     last_lowest_pivot_price > lasrsecond_lowest_pivot_price and \
     lasrsecond_lowest_pivot_price > lastthird_lowest_pivot_price):
    return False
  
   # 两边时间间隔
  if 1 < abs(low_pivots_index[-1] - high_pivots_index[-1]) and  abs(low_pivots_index[-1] - high_pivots_index[-1]) < 15 and \
     1 < abs(low_pivots_index[-2] - high_pivots_index[-1]) and  abs(low_pivots_index[-2] - high_pivots_index[-1]) < 15 and \
     1 < abs(low_pivots_index[-2] - high_pivots_index[-2]) and  abs(low_pivots_index[-2] - high_pivots_index[-2]) < 15 and \
     1 < abs(low_pivots_index[-3] - high_pivots_index[-2]) and  abs(low_pivots_index[-3] - high_pivots_index[-2]) < 15:

    return True
    
  return False
  

  '''
low wave测试集合


'''
high_wave_test = [                 
                 ["603787","20240809", "20241022"],
                 ]
low_wave_test = [
  ["603787","20240207", "20240408"],
  ["603787","20240207", "20240428"]
]


def test_low_wave():
  pass

def test_high_wave():
  pass



if __name__ == "__main__":
  pickle_path = '/home/yao/workspace/Stock/auto_filter/sec_data/daily.pickle' 
  df_dict = data_utils.LoadPickleData(pickle_path)
  for code, val in tqdm(df_dict.items()):
    # if  "000002" not in code:
      # continue
    
    end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
    # end_day_str = "2024-10-24"
    # end_day = dt.datetime.strptime(end_day_str, '%Y-%m-%d').date()

    start_date_str = '2024-02-10'
    start_day = dt.datetime.strptime(start_date_str, '%Y-%m-%d').date()
    
    val.set_index('日期', inplace=True)
    src_daily = val.loc[start_day:]
    src_daily.reset_index(inplace=True)

    df_daily = val.loc[start_day:end_day]
    # print(df_daily.tail())
    df_daily.reset_index(inplace=True)
    
    X = df_daily["收盘"]
    
    data = np.asarray(X)    
    # 

    pivots_high_wave = tech_base.get_pivots(data, 0.15, 0.08)
    pivots_low_wave = tech_base.get_pivots(data, 0.096, 0.05)
    
    # print(pivots)
    # print(data[list(pivots.keys())])
    sel1 = False
    sel1 = filter_high_wave(df_daily, pivots_high_wave, code)
    sel2 = False
    # sel2 =  filter_low_wave(df_daily, pivots_low_wave, code)
    # plt.clf()
    # plot_pivots(data, pivots_low_wave)
    # plot_pivot_line(data, pivots_low_wave)
    # plt.show()
    #TODO 显示比例修改
    if sel1 or sel2:
      data_utils.show_stock_data_eastmoney(code, src_daily,end_day-dt.timedelta(days=88), end_day+dt.timedelta(days=40), vline_data=[])
      # break
      # plt.show()
