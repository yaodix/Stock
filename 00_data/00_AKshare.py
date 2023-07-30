'''
# 20230704
# python 3.8+
AKshare股票下载示例
'''

import akshare as ak
import numpy as np
import pandas as pd
import datetime as dt
from  tqdm import tqdm

import matplotlib.pyplot as plt


def get_sh_sz_A_name():
# 获取所有深A,沪A股市代码,过滤ST，新股、次新股
# 
  list = []

  #获取全部股票代码
  stock_info_a_code_name_df = ak.stock_info_a_code_name()
  total_codes = stock_info_a_code_name_df['code'].tolist()

          
  #非科创板、非创业板、非北京
  for code in total_codes:
      if code[:2] == '60' or code[:1] == '0':
          list.append(code)

  # 非退市
  stock_stop_sh = ak.stock_zh_a_stop_em()
  sh_del = ak.stock_info_sh_delist()
  sz_del = ak.stock_info_sz_delist()
  
  # print(sh_del.head())
  # print(sz_del.head())
  # print(stock_stop_sh.head())
  stop_list = sh_del['公司代码'].tolist() + stock_stop_sh['代码'].tolist()
  # print('000038' in stop_list)
  for code in stop_list:
      if code in list and code in stop_list:
          list.remove(code)
          
  #非ST
  stock_zh_a_st_em_df = ak.stock_zh_a_st_em()
  ST_list = stock_zh_a_st_em_df['代码'].tolist()
  for code in ST_list:
      if code in list and code in ST_list:
          list.remove(code)

  #非次新股、新股，新股数据量小
  stock_zh_a_new_em_df = ak.stock_zh_a_new_em()
  new_list = stock_zh_a_new_em_df['代码'].tolist()
  for code in new_list:
      if code in list :
          list.remove(code)

  stock_zh_a_new_df = ak.stock_zh_a_new()
  new_list = stock_zh_a_new_df['code'].tolist()
  for code in new_list:
      if code in list :
          list.remove(code)
  
  df = stock_info_a_code_name_df[stock_info_a_code_name_df.code.isin(list)]
  
  return df     
        


def get_security_info(symbol = str):
    # 
    return

    
def is_break_low(symbol = str, ratio = 0.6, all_days = 250*5, long_time_days = 250, short_time_days= 150):
  '''
  
  '''
  stocks = get_sh_sz_A_name()
  
  for code in tqdm(stocks.code.tolist()):
    end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
    days = long_time_days * 7 / 5
    #考虑到周六日非交易
    start_day = end_day - dt.timedelta(days)

    start_day = start_day.strftime("%Y%m%d")
    end_day = end_day.strftime("%Y%m%d")   
    
    df_daily = ak.stock_zh_a_hist(symbol=code, period = "daily", start_date=start_day, end_date= end_day, adjust= 'qfq')
    
    # max loc
    # print(f'df_daily {df_daily.head()}')
    close = df_daily.收盘
    # print(f'close {close.head()}')
    
    max_price_index = close.idxmax()
    max_price = close.max()
    
    min_price_index_after_max = close[max_price_index:].idxmin()
    min_price_after_max = close[max_price_index:].min()
    
    # 低价最近发生
    if min_price_after_max < 2:  # 即将退市股票
      continue
    
    if min_price_index_after_max > close.size-10 and  min_price_index_after_max > max_price_index and min_price_after_max / max_price < ratio:
      print(code)
      print(f"max price {max_price}, data {df_daily.loc[max_price_index]['日期']}")
      print(f"min price {min_price_after_max}, data {df_daily.loc[min_price_index_after_max]['日期']}")
      # break
    
    # min loc after max loc
    
  return


def yaogu(decrease_years = 2.5, increse_day = 12):
  stocks = get_sh_sz_A_name()
  output_cnt = 0
  print(f"stock size {len(stocks)}")

  for code in tqdm(stocks.code.tolist()[0:]):
    # print(f"calc stock id {code}")
    end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
    days = 250 * 7 / 5
    #考虑到周六日非交易
    start_day = end_day - dt.timedelta(decrease_years * days)

    start_day = start_day.strftime("%Y%m%d")
    end_day = end_day.strftime("%Y%m%d")   
    
    df_daily = ak.stock_zh_a_hist(symbol=code, period = "daily", start_date=start_day, end_date= end_day, adjust= 'qfq')
    
    # max loc
    # print(f'df_daily {df_daily.head()}')
    close = df_daily.收盘
    # print(f'close {close.head()}')
    # 最近10天涨幅20-35%
    close_prices = close.to_numpy()

    short_max_price = close_prices[-5:].max()
    short_min_price = close_prices[-11:-5].min()
    Inc_ratio = (short_max_price - short_min_price) / short_min_price

    max_price = close_prices[0:125].max()
    min_price = close_prices[-80:-10].min()
    dec_ratio = (max_price - min_price) / max_price
    min_price = close.min()

    if min_price > 1.0 and 0.20 < Inc_ratio and Inc_ratio < 0.35 and  0.4 < dec_ratio :
      print(f"yaogu {code}")
    else :
      output_cnt += 1
      # if (output_cnt %100 == 0):
        # print(f"out code")

def diweiqidong(ratio = 0.7, all_days = 250*5, long_time_days = 250*3, short_time_days= 20*4):
  '''
  启动信号
    1. 2次涨停(或6%以上大阳线)，但股价没有大幅涨起来，青山纸业，间隔3个月60天交易日，有2次涨停
    2. 低位波动涨跌一次，到低位

  test: 青山纸业 600103   20230302-20230529
  test: 学大教育 000526   20230130-20230428
  '''
  stocks = get_sh_sz_A_name()
  
  for code in tqdm(stocks.code.tolist()):
    # code = "600103"
    end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
    
    #考虑到周六日非交易
    days = long_time_days * 7 / 5
    start_day = end_day - dt.timedelta(days)

    start_day = start_day.strftime("%Y%m%d")
    end_day = end_day.strftime("%Y%m%d")   
    # end_day = "20230529"
    
    df_daily = ak.stock_zh_a_hist(symbol=code, period = "daily", start_date=start_day, end_date= end_day, adjust= 'qfq')
    
    # max loc
    # print(f'df_daily {df_daily.tail()}')
    close = df_daily.收盘
    increase = df_daily.涨跌幅
    # print(f'close {close.head()}')
    
    max_price_index = close[:-short_time_days*2].idxmax()
    max_price = close[:-short_time_days*2].max()
    
    min_price_near_index = close[-short_time_days*2:].idxmin()
    min_price_near = close[-short_time_days*2:].min()
    # 低价最近发生
    if min_price_near < 1:  # 即将退市股票
      continue
    
    # 双响炮
    # 最近15天内有涨幅
    break_2_index = increase[-15:-5].idxmax()
    break_2_ratio = increase[-15:-5].max()
    inc_2_val = close[break_2_index]

    break_1_f = increase[-short_time_days:break_2_index]
    break_1_index = increase[-short_time_days:break_2_index].idxmax()
    break_1_ratio = increase[-short_time_days:break_2_index:].max()
    inc_1_val = close[break_1_index]


  # 最近10天涨幅超过6，只有2个
    cnt = np.sum(increase[-10:]>=6)  
    # 涨停后回调
    cnt2 = np.sum(close[-5:] < inc_2_val) 

    # 两个
  
  # 最近价格都小于

    if break_1_ratio > 8 and break_2_ratio > 8 and abs(inc_2_val - inc_1_val)/ inc_1_val < 0.2 and abs(inc_2_val - inc_1_val)/ inc_2_val < 0.2 and break_2_index - break_1_index > 20 and \
       min_price_near_index > max_price_index and min_price_near / max_price < ratio and  min(inc_2_val, min_price_near) / max(inc_2_val ,min_price_near) > 0.65 and \
        cnt <=2  and cnt2 > 3:
      print(code)
      print(f"break 1 {break_1_ratio}, data {df_daily.loc[break_1_index]['日期']}")
      print(f"break 2 {break_2_ratio}, data {df_daily.loc[break_2_index]['日期']}")


      # break
    # 与之前10年股价差排序

    # 低位横盘长度排序
    # break
    
  return



if __name__ == "__main__":
    # n = get_sh_sz_A_name()
    # print(n.head)
    # res = is_break_low()
    # yaogu()
    diweiqidong()

