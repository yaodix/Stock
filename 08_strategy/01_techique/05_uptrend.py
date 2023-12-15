
# 01_basic/02_my_zigzag.py
import sys
sys.path.append(r"/home/yao/myproject/Stock/01_basic")
sys.path.append(r"/home/yao/myproject/Stock/00_data")
import akshare as ak
import numpy as np
import pandas as pd
import datetime as dt
from  tqdm import tqdm

import matplotlib.pyplot as plt


from my_zigzag import my_zigzag, plot_pivots
from stock_data_utils import get_sh_sz_A_name


## 低位，有10个点以上的反弹回调，再反弹次数幅度大于5%，有2个大于3个点的涨幅
long_time_days = 250
stocks = get_sh_sz_A_name()
for code in tqdm(stocks.code.tolist()):
  # print(code)
  # code = "000088"
  end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
  days = long_time_days * 7 / 5
  #考虑到周六日非交易
  start_day = end_day - dt.timedelta(days)

  start_day = start_day.strftime("%Y%m%d")
  end_day = end_day.strftime("%Y%m%d")   
  
  df_daily = ak.stock_zh_a_hist(symbol=code, period = "daily", start_date=start_day, end_date= end_day, adjust= 'qfq')

  # df_daily = ak.stock_zh_a_hist(symbol="601225", period = "daily", start_date= "20221100", end_date="20230221")
  X = df_daily["收盘"]

  data = np.asarray(X)

  pivots = my_zigzag(data)
  if pivots.__len__() < 4:
    continue
  
  indice =[]
  for key, val in pivots.items():
    indice.append(key)

  indice = np.asarray(indice)
  pre_low_val = indice[-3]
  last_high_val = indice[-2]

  last_low_val = indice[-1]
  rev_point = {}
  if last_high_val > pre_low_val and last_high_val > last_low_val:
    # 天数不能太久
    if indice[-2] - indice[-3] < 20 and indice[-1] - indice[-2] < 20:
      # 最近波动5个点
      tail_data = data[indice[-2]:-1]
      pivots_tail = my_zigzag(tail_data, 0.05)
      ind_tail =[]
      for key, val in pivots_tail.items():
        ind_tail.append(key)
      if pivots_tail.__len__() == 3 and pivots_tail[ind_tail[1]] < pivots_tail[-1]:
        print("hello")
        print(code)

        rev_point[ind_tail[1]] = 0
        

  # plot_pivots(X, pivots)
  # plt.show()
