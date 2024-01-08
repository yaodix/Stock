

# 慢牛波动
# 快牛波动不可持续
# 慢牛股

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
from stock_data_utils import get_sh_sz_A_name, moving_average

long_time_days = 250
win = 10  # 求均值win
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
  
  # df_daily = ak.stock_zh_a_hist(symbol=code, period = "daily", start_date=start_day, end_date= end_day, adjust= 'qfq')

  df_daily = ak.stock_zh_a_hist(symbol="600985", period = "daily", start_date = start_day, end_date = end_day)
  X = df_daily["收盘"]

  data = np.asarray(X)

  
  # tech anaysis
  #########
  check_data = data[-40:]  # 2个月
  ma5 = moving_average(check_data, win)
  greater_cnt = np.sum(check_data > ma5)
  if greater_cnt > 4:
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
