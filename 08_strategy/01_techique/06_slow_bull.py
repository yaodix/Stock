
## 中高位波动
# 股价比高位略低，反复波动，且一定要有涨停出现

# 慢牛波动
# 快牛波动不可持续
# 慢牛股

import sys
sys.path.append(r"/home/yao/myproject/Stock/01_basic")
sys.path.append(r"/home/yao/myproject/Stock/00_data")
import akshare as ak
import numpy as np
import pandas as pd
import datetime as dt
from  tqdm import tqdm

import matplotlib.pyplot as plt


from my_zigzag import get_wave, plot_pivots
from stock_data_utils import get_sh_sz_A_name, moving_average

long_time_days = 250
win = 20  # 求均值win
increase_bull = []
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

  # df_daily = ak.stock_zh_a_hist(symbol="605599", period = "daily", start_date = start_day, end_date = end_day)
  X = df_daily["收盘"]

  data = np.asarray(X)

  pivots = get_wave(data)
  if pivots.__len__() < 2:
    continue
  
  # tech anaysis
  #########catch third wave
  if pivots.__len__() > 1:
    wave_1_start = list(pivots.keys())[-2]
    wave_1_end = list(pivots.keys())[-1]
    increase_data = data[wave_1_start : wave_1_end]
    
    diff = np.diff(increase_data)
    diff = np.insert(diff, 0,0)
    ratio_data = diff / increase_data
    max_ratio = np.max(ratio_data)
    min_ratio = np.min(ratio_data)

    if (data[wave_1_end] > data[wave_1_start] and (data[wave_1_end] - data[wave_1_start])/data[wave_1_start] > 0.2 and
        wave_1_end - wave_1_start > 20 * 2 and  # 长超过2个月
        max_ratio < 0.06 and min_ratio > -0.06
        ):
      
        # above mav line 20 or 30      
        # ma20 = moving_average(increase_data_pre_win, win)
        # greater_cnt = np.sum(increase_data[0:win/2] > increase_data_pre_win[win: win/2])
        if greater_cnt > 2:
          continue
        
        # ignore low asset < 50 e
        stock_individual_info_em_df = ak.stock_individual_info_em(symbol=code)
        shizhi = stock_individual_info_em_df["value"][0]
        if shizhi < 4500000000: # 45亿
          continue
        # ignore high price recent year
        
        increase_bull.append(code)
        

for c in increase_bull:
  print(f"code {c}")


        

  # plot_pivots(X, pivots)
  # plt.show()
