# 20230704
# python 3.8+

import akshare as ak
import numpy as np
import pandas as pd
import datetime as dt

import matplotlib.pyplot as plt


# 下载A股所有数据
# all_df = ak.stock_zh_a_spot_em()

#  stock_sh_a_spot_em  # 沪A
#  stock_sz_a_spot_em  # 深A
 
#  stock_individual_info_em    # 个股信息查询
 
# print(all_df.head())

# 下载个股日k数据图
# 20230131-20230531,　形态, 83周期
# target: 600640，20230303 - 20030705

def test_one():
  df_daily = ak.stock_zh_a_hist(symbol="000338", period = "daily", start_date= "20230131", end_date="20230531")
  t_df_daily = ak.stock_zh_a_hist(symbol="600640", period = "daily", start_date= "20230302", end_date="20230704")
  print(df_daily.tail())

  X = df_daily["收盘"]
  print(X.tail())

def get_sh_sz_A_name():
# 获取所有深A,沪A股市代码,过滤ST，新股、次新股
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
        

def test_merge():
  df1 = pd.DataFrame(
    {
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
        "C": ["C0", "C1", "C2", "C3"],
        "D": ["D0", "D1", "D2", "D3"],
    },
    index=[0, 1, 2, 3],)


  df2 = pd.DataFrame(
      {
          "A": ["A4", "A5", "A6", "A7"],
          "B": ["B4", "B5", "B6", "B7"],
          "C": ["C4", "C5", "C6", "C7"],
          "D": ["D4", "D5", "D6", "D7"],
      },
    index=[0, 1, 2, 3],)

  frames = [df1, df2]

  result = pd.concat(frames, ignore_index=True)
  print(f'merge {result}')
  

def get_security_info(symbol = str):
    # 
    return

    
def is_break_low(symbol = str, ratio = 0.6, all_days = 250*5, long_time_days = 250, short_time_days= 150):
  '''
  
  '''
  stocks = get_sh_sz_A_name()
  
  for code in stocks.code.tolist():
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

if __name__ == "__main__":
    # n = get_sh_sz_A_name()
    # print(n.head)
    res = is_break_low()

