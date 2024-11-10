
import akshare as ak
import datetime as dt
import pandas as pd
import numpy as np
from  tqdm import tqdm
import os
import pickle
import mplfinance as mpf

import matplotlib.pyplot as plt


def GetSecurityCode():
  '''
  func:get all sh sz sec code, filter ST, new stock, secod new stock
  '''
  list = []

  # get all stock code
  stock_info_a_code_name_df = ak.stock_info_a_code_name()
  total_codes = stock_info_a_code_name_df['code'].tolist()
          
  # exclude kechuang, beijing
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


def dump(security_pool, pickle_file, period_unit, years = 10):
  '''
  func: dump data to pickle file
  period: choice of {'daily', 'weekly', 'monthly'}
  '''
  pool = []
  if isinstance(security_pool, list):
    pool = security_pool
  else:
    pool = security_pool.code.tolist()
  
  df_dict = {}
  days = years * 365
  end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
  start_day = end_day - dt.timedelta(days)
  print(f"start day: {start_day}, end day: {end_day}")
  end_day = end_day.strftime("%Y%m%d")   
  start_day = start_day.strftime("%Y%m%d")

  for code in tqdm(pool):    
    df = ak.stock_zh_a_hist(symbol=code, period = period_unit, start_date=start_day, end_date= end_day, adjust= 'qfq')
    # df.rename(columns={
    # '日期': 'Date',
    # '开盘': 'Open',
    # '收盘': 'Close',
    # '最高': 'High',
    # '最低': 'Low',
    # '成交量': 'Volume'
    # },inplace=True)
    df_dict[code] = df
    # break
    
  with open(pickle_file, 'wb') as handle:
    pickle.dump(df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def LoadPickleData(pickle_path):
  if not os.path.exists(pickle_path):
    print("no file " + pickle_path)
    return
    
  with open(pickle_path, 'rb') as handle:
      df_dict = pickle.load(handle) 
  
  return df_dict

def isTradeDay(trade_date=''):
  '''
  func: check if trade day
  '''
  if trade_date == '':
    trade_date = dt.date.today().strftime("%Y%m%d")
  df = ak.stock_zh_a_hist(symbol="000001", period = 'daily', start_date=trade_date, end_date= trade_date, adjust= 'qfq')
  if df.empty:
    return False
  else:
    return True

def show_stock_data_eastmoney(code, df_one, start_date="", end_date="", vline_data = []):
  '''
    vline_data:['2024-08-23']
  '''

  if start_date == "":
    start_date = dt.date.today() - dt.timedelta(days=50)
    start_date = start_date.strftime("%Y%m%d")

  if end_date == "":
    end_date = dt.date.today().strftime("%Y%m%d")
  # 将日期列设置为索引，并转换为 datetime 类型
  df_one['日期'] = pd.to_datetime(df_one['日期'])

  # 调整 DataFrame 列名以符合 mplfinance 的要求
  df_show = df_one.rename(columns={
    '日期': 'Date',
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

def updateToLatestDay(pickle_file, period_unit):
  '''
  update data to latest day
  '''
  df_dict = LoadPickleData(pickle_file)
  first_df = next(iter(df_dict.values()))
  last_date = first_df['日期'].iloc[-1]
  last_date = last_date + dt.timedelta(days=1)
  cur_data = dt.date.today()
  last_date_str = last_date.strftime("%Y%m%d")
  cur_data_str = cur_data.strftime("%Y%m%d")   
  if not isTradeDay(last_date_str) and cur_data-last_date < dt.timedelta(days=2):
    print(f"no need to update data")
    return df_dict
  
  else:
    print(f"update data to today, last day {last_date}")  
    for code, df in tqdm(df_dict.items()):
      add_df = ak.stock_zh_a_hist(symbol=code, period = period_unit, start_date=last_date_str, end_date= cur_data_str, adjust= 'qfq')
      df = df.append(add_df, ignore_index=True)
      df_dict[code] = df
      # break

    with open(pickle_file, 'wb') as handle:
      pickle.dump(df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
    return df_dict
     

if __name__ == '__main__':
  monthly_path = './sec_data/monthly.pickle' 
  weekly_path = './sec_data/weekly.pickle'
  daily_path = './sec_data/daily.pickle'
  # df = GetSecurityCode()  
  # dump(df, daily_path,'daily', 10)
  # dump(df, weekly_path,'weekly', 50)
  # dump(df, monthly_path,'monthly', 50)
