import akshare as ak
import numpy as np
import pandas as pd
import datetime as dt
from  tqdm import tqdm
import mplfinance as mpf

import matplotlib.pyplot as plt
import os
import pickle

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


def LoadPickleData(pickle_path):
  if not os.path.exists(pickle_path):
    print("no file " + pickle_path)
    return
    
  with open(pickle_path, 'rb') as handle:
      df_dict = pickle.load(handle) 
  
  return df_dict

def LoadData(pickle_path):
  if not os.path.exists(pickle_path):
    print("no file " + pickle_path)
    return
    
  with open(pickle_path, 'rb') as handle:
      df_dict = pickle.load(handle) 
  
  return df_dict

def show_stock_data_eastmoney(code, df_one, start_date="20200630", end_date="20240530", vline_data = ['2024-08-23']):
  # 将Data列设置为索引，并转换为 datetime 类型

  df_one['Date'] = pd.to_datetime(df_one['Date'])

  # 调整 DataFrame 列名以符合 mplfinance 的要求
  df_show = df_one.rename(columns={
    'Date': 'Date',
    '开盘': 'Open',
    '收盘': 'Close',
    '最高': 'High',
    '最低': 'Low',
    '成交量': 'Volume'
  })
  df_show.set_index('Date', inplace=True)
  df_show = df_show.loc[start_date:end_date]


  # 定义 mplfinance 的自定义风格
  mc = mpf.make_marketcolors(up='r', down='g', volume='inherit')
  s = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc) # 

  # 使用 mplfinance 绘制 K 线图，并应用自定义风格
  fig_name = "./workdata/" + code+".png"
  mpf.plot(df_show, type='candle', style=s,
       title=f"{code} K linechart",
       ylabel='Price',
       ylabel_lower='Vol',
       volume=True,
       vlines=dict(vlines=vline_data,linewidths=(1,)),
      #  mav=(5,20,250),
       show_nontrading=False,
       savefig=dict(fname=fig_name,dpi=100,pad_inches=0.25)
       )
  
  # mpf.show()