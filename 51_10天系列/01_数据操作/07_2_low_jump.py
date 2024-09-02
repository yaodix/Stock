
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


def CheckBreakUp(df_daily):
  # 最后8天内有涨停板
  last_idx = 15
  close_price = np.asarray(df_daily["收盘"])[-last_idx-1:]
  diff = np.diff(close_price)
  diff = diff / close_price[:-1]
  close_price = close_price[1:]

  daily_limit_idx = np.where(diff > 0.095)[0]
  # print(type(daily_limit_idx))
  if (daily_limit_idx.shape[0] < 1 or daily_limit_idx.shape[0] > 2):
    return False
  
  # 2个涨停不能太近
  if (daily_limit_idx.size == 2):
    if (daily_limit_idx[0] - daily_limit_idx[1] <3):
      return False
    
  # 最新涨停
  if diff.__len__() -  daily_limit_idx[-1] < 3:
    return False
  
  # 涨停后的涨幅较小
  arr = diff[daily_limit_idx[-1]+1:]
  if np.any(abs(arr) > 0.06):
    return False
  
  # 除涨停外，幅度都小
  arr2 = diff[:daily_limit_idx[-1]]
  if np.any(abs(arr2) > 0.05):
    return False
  
  # 涨停后有价格低于涨停前一天价格
  if np.any(close_price[daily_limit_idx[-1]+1:] < close_price[daily_limit_idx[-1]-1]):
    return False
  
  # 涨停后跌幅大小
  if close_price[daily_limit_idx[-1]] <  close_price[-1]:
    return False
  
  rr = abs(close_price[daily_limit_idx[-1]] - close_price[-1])/ close_price[daily_limit_idx[-1]]
  if abs(close_price[daily_limit_idx[-1]] - close_price[-1])/ close_price[daily_limit_idx[-1]] > 0.06:
    return True

  # return True



if __name__ == "__main__":
  pickle_path = '/home/yao/workspace/Stock/51_10天系列/01_数据操作/df_0830.pickle' 
  df_dict = LoadPickleData(pickle_path)
  for code, val in tqdm(df_dict.items()):
    if code < "000557":
      continue
    # val.drop([len(val)-1],inplace=True)

    end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
    end_day = end_day.strftime("%Y%m%d")   
    start_date_str = '01-01-2024'
    start_day = dt.datetime.strptime(start_date_str, '%m-%d-%Y').date()
    # val = ak.stock_zh_a_hist(symbol=code, start_date=start_day, end_date="20241207" ,period = "daily", adjust= 'qfq')
    # val = ak.stock_zh_a_hist(symbol=code, start_date=start_day, end_date="20241207", period = "weekly", adjust= 'qfq')
    # print(val.tail(5))
    
    
    df_daily = val[val["日期"]> start_day]
    close_price = np.asarray(df_daily["收盘"])
    # print(pivots)
    # print(data[list(pivots.keys())])
    sel = CheckBreakUp(df_daily)
    # 进一步筛选
    # sel = True
    #TODO 显示比例修改
    # break
    if sel:
      # tobuy = raise_long_buy(data, pivots)
      # if not tobuy:
        # continue
      print(code)
      show_stock_data_eastmoney(code, df_daily,"20240801", end_day)
      # break
      # plt.show()
