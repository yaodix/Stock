
import sys
import os
import numpy as np
from  tqdm import tqdm
import datetime as dt

cur_file = os.path.abspath(__file__)
start_index = cur_file.find("auto_filter")
pro_path = cur_file[:start_index+11]
sys.path.append(pro_path)

import tech.tech_base as tech_base
import data_utils

def raiseLimitTwoImpl(df_daily, code, idx_reverse):
  '''
   two raiselimit and decade recently, but price not raise too much
  '''
  last_idx = 30
  raise_ratio_exp = 0.07
  raise_limit = 0.096
  
  if ("30" in code[0:2]):
    raise_ratio_exp = 0.09
    raise_limit = 0.198

  close_price = np.asarray(df_daily["Close"])[-last_idx-1:]
  daily_limit_idx, idx_r, diff_ratio = tech_base.get_daily_raise_limit(close_price, code)
  # print(f" {code} {daily_limit_idx.size}")
  if daily_limit_idx.size < 2:
    return False
  daily_limit_idx = daily_limit_idx[-2:]
  idx_r = idx_r[-2:]
  # 2个涨停不能太近
  if (daily_limit_idx[1] - daily_limit_idx[0] < 3):
    return False
    
  # 最新涨停
  if close_price.__len__() -  daily_limit_idx[-1] < 3:
    return False
  
  # 两次涨停的价格，变动过大
  if abs(close_price[daily_limit_idx[0]] - close_price[daily_limit_idx[1]] ) / close_price[daily_limit_idx[0]] > raise_limit*0.8:
    return False
  
  # 最后一次涨停后的涨幅较小
  arr = diff_ratio[daily_limit_idx[-1]+1:]
  if np.any(abs(arr) > raise_ratio_exp):
    return False
  
  # 2次涨停之间，连续跌幅幅度不大
  gap_diff = diff_ratio[daily_limit_idx[0]+1:daily_limit_idx[-1]]
  sum_two = np.convolve(gap_diff,np.ones(2,dtype=int),'valid')
  if np.any((sum_two) < -0.1):
    return False
  
  # 涨停后有价格低于涨停前一天价格,
  # if np.any(close_price[daily_limit_idx[-1]+1:] < close_price[daily_limit_idx[-1]-1]):
  #   return False
  
  # 涨停后到最新价格跌幅大小
  min_in_two_raise_limt = np.min(close_price[daily_limit_idx[0]:daily_limit_idx[-1]])
  if min_in_two_raise_limt >  close_price[-1]:
    return False
  
  # 涨停后到最新价格跌幅大小
  rr = (close_price[daily_limit_idx[-1]] - close_price[-1])/ close_price[daily_limit_idx[-1]]
  # 两次跌幅的比例相差不大
  fail_1 = (close_price[daily_limit_idx[0]] - close_price[daily_limit_idx[1]-1])/ close_price[daily_limit_idx[0]]
  fail_2 = (close_price[daily_limit_idx[1]] - close_price[-1])/ close_price[daily_limit_idx[1]]
  if rr > 0.04 or abs(fail_1 - fail_2) < 0.02:
    # idx_reverse = idx_r
    return True

  return False

def raiseLimitTwo(df_dict):
  select_dic= {}
  print("task raise limit two")
  for code, df_daily in tqdm(df_dict.items()):
    idx_re = []
    sel = raiseLimitTwoImpl(df_daily, code, idx_re)
    if sel:
      select_dic[code] = idx_re
  return select_dic


test_list = [
             ["000826", "20241101"],
             ["600839", "20240829"],  # sichuanchanghong
  ["603787", "20240408"],
             
             ]

if __name__ == "__main__":
  df_dict = data_utils.LoadPickleData(pro_path+"/sec_data/daily.pickle")

  test_dict = {}
  for ite in test_list:
    test_dict[ite[0]] = df_dict[ite[0]]
    end_day = dt.datetime.date(dt.datetime.strptime(ite[1], "%Y%m%d"))
    test_dict[ite[0]] = test_dict[ite[0]][test_dict[ite[0]]["Date"] <= end_day]
    
  res_dict = raiseLimitTwo(test_dict)
  print(res_dict)
