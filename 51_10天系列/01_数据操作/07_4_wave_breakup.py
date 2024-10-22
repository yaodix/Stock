'''
1, structure clearly, wave
2. rasing limit in wave

test: 
'''


#  斜率K + 单调性 + 支点数量 + 长度?
# find and sort

# filter by slope of pivot

import sys
sys.path.append(r"/home/yao/workspace/Stock/01_basic")
sys.path.append(r"/home/yao/workspace/Stock/00_data")

import akshare as ak
import numpy as np
import matplotlib.pyplot as plt
from  tqdm import tqdm
import datetime as dt

from my_zigzag import get_pivots, plot_pivots, plot_pivot_line, get_daily_raise_limit
from stock_data_utils import get_sh_sz_A_name, LoadPickleData,show_stock_data_eastmoney

def count_numbers_in_range(arr, start, end):
    # 使用条件索引来选择在指定区间内的数字
    mask = (arr >= start) & (arr <= end)
    # 统计满足条件的数字个数
    count = np.count_nonzero(mask)
    return count
def filter_pivot_line_raiselimitinwave(src_data, pivots, security_code, verbose = False):
  
  ll = np.array(src_data["收盘"])[-1]
  raise_limit_idx =  get_daily_raise_limit(src_data["收盘"], security_code)
  
  low_pivots_index = [k for k, v in pivots.items() if v == -1]
  high_pivots_index = [k for k, v in pivots.items() if v == 1]
  if (len(low_pivots_index) < 2):
    return False
  
  if (high_pivots_index[-1] > low_pivots_index[-1]):
    return False
  
  # 涨停数量
  raise_1_start_low_idx = low_pivots_index[-2]
  raise_1_end_high_idx = high_pivots_index[-1]
  count = count_numbers_in_range(raise_limit_idx, raise_1_start_low_idx, raise_1_end_high_idx)
  if count < 1:  
    return False
  
  # 收盘价格与前低点价格对比
  last_price = np.asarray(src_data["收盘"])[-1]
  last_lowest_pivot_price = np.asarray(src_data["收盘"])[low_pivots_index[-1]]
  if last_price > 1.1 * last_lowest_pivot_price :
    return False
  
  # 2个low pivots点价格相差不大, 从高位降落到0.618以下
  last_high_pivot_price = np.asarray(src_data["收盘"])[high_pivots_index[-1]]
  pre_lowest_pivot_price = np.asarray(src_data["收盘"])[low_pivots_index[-2]]
  if       abs(last_lowest_pivot_price - last_high_pivot_price) / abs(last_high_pivot_price-pre_lowest_pivot_price) >= 0.5999:
    return True

  return False
  

if __name__ == "__main__":
  pickle_path = '/home/yao/workspace/Stock/51_10天系列/01_数据操作/df_1022.pickle' 
  df_dict = LoadPickleData(pickle_path)
  for code, val in tqdm(df_dict.items()):
    # if  "000526" not in code:
      # continue
    
    # end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
    end_day_str = "2024-10-22"
    end_day = dt.datetime.strptime(end_day_str, '%Y-%m-%d').date()

    start_date_str = '2024-02-10'
    start_day = dt.datetime.strptime(start_date_str, '%Y-%m-%d').date()
    
    val.set_index('日期', inplace=True)
    df_daily = val.loc[start_day:end_day]
    # print(df_daily.tail())
    df_daily.reset_index(inplace=True)
    
    X = df_daily["收盘"]
    
    data = np.asarray(X)    
    pivots = get_pivots(data, 0.15, 0.08)
    # print(pivots)
    # print(data[list(pivots.keys())])
    sel =  filter_pivot_line_raiselimitinwave(df_daily, pivots, code)
    # sel = True
    #TODO 显示比例修改
    if sel:
      print(code)
      show_stock_data_eastmoney(code, df_daily,"20240501", end_day)
      # break
      # plt.show()
