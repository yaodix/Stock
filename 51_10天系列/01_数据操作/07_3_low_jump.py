
import sys
sys.path.append(r"/home/yao/workspace/Stock/01_basic")
sys.path.append(r"/home/yao/workspace/Stock/00_data")

import akshare as ak
import numpy as np
import matplotlib.pyplot as plt
from  tqdm import tqdm
import datetime as dt

from my_zigzag import get_pivots, plot_pivots, plot_pivot_line
from stock_data_utils import get_sh_sz_A_name, LoadPickleData, show_stock_data_eastmoney


def CheckBreakUp(df_daily, code):
  # 最后8天内有涨停板
  last_idx = 30

  raise_limit = 0.096
  raise_ratio_exp = 0.06
  if ("30" in code[0:2]):
    raise_limit = 0.198
    raise_ratio_exp = 0.08

  close_price = np.asarray(df_daily["收盘"])[-last_idx-1:]
  diff = np.diff(close_price)
  diff = diff / close_price[:-1]
  close_price = close_price[1:]

  daily_limit_idx = []
  
  for idx, val in enumerate(diff[::-1]):
    if val > raise_limit:
      daily_limit_idx.append(last_idx-1 - idx)
      if daily_limit_idx.__len__() == 2:
        break

  if daily_limit_idx.__len__() < 2:
    return False

  daily_limit_idx = np.asarray(daily_limit_idx)
  daily_limit_idx = np.flip(daily_limit_idx)
  # print(f" {code} {daily_limit_idx.size}")
  # 2个涨停不能太近
  if (daily_limit_idx[1] - daily_limit_idx[0] < 3):
    return False
    
  # 最新涨停
  if diff.__len__() -  daily_limit_idx[-1] < 3:
    return False
  
  # 两次涨停的价格，
  if abs(close_price[daily_limit_idx[0]] - close_price[daily_limit_idx[1]] ) / close_price[daily_limit_idx[0]] > raise_limit*0.8:
    return False
  
  # 最后一次涨停后的涨幅较小
  arr = diff[daily_limit_idx[-1]+1:]
  if np.any(abs(arr) > raise_ratio_exp):
    return False
  
  # 涨停之间，幅度都小
  arr2 = diff[daily_limit_idx[0]+1:daily_limit_idx[-1]]
  if np.any(abs(arr2) > raise_ratio_exp):
    return False
  
  # 涨停后有价格低于涨停前一天价格
  if np.any(close_price[daily_limit_idx[-1]+1:] < close_price[daily_limit_idx[-1]-1]):
    return False
  
  # 涨停后到最新价格跌幅大小
  min_in_two_raise_limt = np.min(close_price[daily_limit_idx[0]:daily_limit_idx[-1]])
  if min_in_two_raise_limt >  close_price[-1]:
    return False
  
  # rr = abs(close_price[daily_limit_idx[-1]] - close_price[-1])/ close_price[daily_limit_idx[-1]]
  # if abs(close_price[daily_limit_idx[-1]] - close_price[-1])/ close_price[daily_limit_idx[-1]] > raise_ratio_exp:
  #   return True

  return True



if __name__ == "__main__":
  pickle_path = '/home/yao/workspace/Stock/51_10天系列/01_数据操作/df_0908.pickle' 
  df_dict = LoadPickleData(pickle_path)
  for code, val in tqdm(df_dict.items()):
    if code < "000521":
      continue
    # val.drop([len(val)-1],inplace=True)

    dt_end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
    end_day = dt_end_day.strftime("%Y%m%d")   
    start_date_str = '01-01-2024'
    start_day = dt.datetime.strptime(start_date_str, '%m-%d-%Y').date()
    # val = ak.stock_zh_a_hist(symbol=code, start_date=start_day, end_date="20241207" ,period = "daily", adjust= 'qfq')
    # val = ak.stock_zh_a_hist(symbol=code, start_date=start_day, end_date="20241207", period = "weekly", adjust= 'qfq')
    # print(val.tail(5))
    
    
    df_daily = val[val["日期"]> start_day]
    close_price = np.asarray(df_daily["收盘"])
    # print(pivots)
    # print(data[list(pivots.keys())])
    sel = CheckBreakUp(df_daily, code)
    # 进一步筛选
    # sel = True
    #TODO 显示比例修改
    # break
    if sel:
      # tobuy = raise_long_buy(data, pivots)
      # if not tobuy:
        # continue
      print(code)
      plot_start = dt_end_day - dt.timedelta(days=40)
      show_stock_data_eastmoney(code, df_daily,plot_start, end_day)
      # break
      # plt.show()
