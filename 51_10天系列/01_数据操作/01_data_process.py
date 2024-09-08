
'''
筛选股票和保存股票数据
'''
import akshare as ak
import os
import numpy as np
import pandas as pd
import datetime as dt
from  tqdm import tqdm
import pickle

import matplotlib.pyplot as plt

# get all security code

def GetSecurityCode():
  '''
  获取所有深A,沪A股市代码,过滤ST、新股、次新股
  '''
  list = []

  #获取全部股票代码
  stock_info_a_code_name_df = ak.stock_info_a_code_name()
  total_codes = stock_info_a_code_name_df['code'].tolist()

          
  #非科创板、非北京
  for code in total_codes:
      if code[:2] == '60' or code[:1] == '0' or code[:2] == '30':
          list.append(code)

  # 非退市
  stock_stop_sh = ak.stock_zh_a_stop_em()
  sh_del = ak.stock_info_sh_delist()
  sz_del = ak.stock_info_sz_delist()
  
  stop_list = sh_del['公司代码'].tolist() + stock_stop_sh['代码'].tolist()
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

# 下载指定所有
def dump(security_pool, pickle_file, years = 100):
  pool = []
  if isinstance(security_pool, list):
    pool = security_pool
  else:
    pool = security_pool.code.tolist()
  
  df_dict = {}
  for code in tqdm(pool):
    end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
    days = years * 365
    start_day = end_day - dt.timedelta(days)
    start_day = start_day.strftime("%Y%m%d")
    end_day = end_day.strftime("%Y%m%d")   
    
    df = ak.stock_zh_a_hist(symbol=code, period = "daily", start_date=start_day, end_date= end_day, adjust= 'qfq')
    # df = ak.stock_zh_a_hist(symbol=code, period = "weekly", start_date=start_day, end_date= end_day, adjust= 'qfq')
    
    df_dict[code] = df
    
  with open(pickle_file, 'wb') as handle:
    pickle.dump(df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  pickle_path = './df_0908.pickle' 
  df = GetSecurityCode()  
  dump(df, pickle_path)
