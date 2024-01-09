
# 01_basic/02_my_zigzag.py
import sys
sys.path.append(r"/home/yao/myproject/Stock/01_basic")
sys.path.append(r"/home/yao/myproject/Stock/00_data")
# sys.path.append(r"/home/yao/workspace/Stock/01_basic")
# sys.path.append(r"/home/yao/workspace/Stock/00_data")
import akshare as ak
import numpy as np
import pandas as pd
import datetime as dt
from  tqdm import tqdm

import matplotlib.pyplot as plt


from my_zigzag import get_wave, plot_pivots
from stock_data_utils import get_sh_sz_A_name


long_time_days = 250
uptrend_code = []
stocks = get_sh_sz_A_name()
for code in tqdm(stocks.code.tolist()):
  # print(code)
  # code = "601595"
  end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
  days = long_time_days * 7 / 5
  #考虑到周六日非交易
  start_day = end_day - dt.timedelta(days)

  start_day = start_day.strftime("%Y%m%d")
  end_day = end_day.strftime("%Y%m%d")   
  
  df_daily = ak.stock_zh_a_hist(symbol=code, period = "daily", start_date=start_day, end_date= end_day, adjust= 'qfq')

  # df_daily = ak.stock_zh_a_hist(symbol=code, period = "daily", start_date = "20230101", end_date = "20231201")
  X = df_daily["收盘"]

  data = np.asarray(X)

  pivots = get_wave(data)
  if pivots.__len__() < 2:
    continue
  
  # tech anaysis
  #########catch second wave
  if pivots.__len__() > 4:
    wave_1_start = list(pivots.keys())[-4]
    wave_1_end = list(pivots.keys())[-3]
    wave_2_start = list(pivots.keys())[-2]
    wave_2_end = list(pivots.keys())[-1]

    if pivots[wave_2_end] != -1: # 最后一个必须为价格低点
      continue

    if (data.__len__() - wave_2_end >= 2 and
        data[wave_2_start] < data[wave_1_start] and data[wave_2_end] > data[wave_1_end] and 
        5 <  wave_2_end - wave_1_end and  wave_2_end - wave_1_end < 60 and 
        10 <  wave_1_end - wave_1_start):  # 时间周期

        if (data[wave_2_end] - data[wave_1_end])/data[wave_1_end] < 0.08: # second wave start is higher
          continue
        
        
        max_value = np.max(data[0:wave_1_end])
        if (max_value- data[wave_1_end]) / max_value > 0.45:  # 跌了多少
          # wave should not too 
          # if  (0.1 < (data[wave_2_start] - data[wave_1_end]) / data[wave_1_end] and 
          #             (data[wave_2_start] - data[wave_1_end]) / data[wave_1_end] < 0.6) :
          #   continue
          # ignore high stock price
          if data[wave_2_end] > 60:
            continue
          
          # ignore low asset < 50 e
          stock_individual_info_em_df = ak.stock_individual_info_em(symbol=code)
          shizhi = stock_individual_info_em_df["value"][0]
          if shizhi < 4500000000: # 45亿
            continue
          # ignore high price recent year
          
          uptrend_code.append(code)
          # print(f"append {code}")
          # break
        

# 如果确立趋势，找到走势最强，涨势最好的--sort  
for c in uptrend_code:
  print(f"code {c}")


        

  # plot_pivots(X, pivots)
  # plt.show()
