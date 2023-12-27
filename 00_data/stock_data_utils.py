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


def moving_average(x, w):
    '''
    w = 5, 5日均线
    '''
    tmp = np.convolve(x, np.ones(w), 'same') / w
    half_w = int(w/2)
    tmp[:half_w] = x[:half_w]
    tmp[-half_w:] = x[-half_w:]
    return tmp