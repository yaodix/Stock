
import akshare as ak
import datetime as dt
import pandas as pd
import numpy as np
from  tqdm import tqdm
import os
import pickle
import mplfinance as mpf
from scipy import interpolate

import matplotlib.pyplot as plt
from ./tech/board_class import LoadSwClassDict
import warnings
warnings.filterwarnings("ignore")


def GetSecurityCode():
  '''
  func:get all sh sz sec code, filter ST, new stock, secod new stock
  ret: df of [code, name]
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

def getMarketCapDict(code_list):
  code_cap_dict = {}
  print(f"get market cap")
  for code in tqdm(code_list):
    stock_individual_info_em_df = ak.stock_individual_info_em(symbol=code)
    cap = stock_individual_info_em_df.iloc[4]["value"]
    code_cap_dict[code] = cap
  return code_cap_dict

def industryLeaderSW():
  '''
  sw class leader, class: code
  '''
  thirdclass_code_dict,_,_ = LoadSwClassDict()
  all_code =  [code for codes in thirdclass_code_dict.values() for code in codes]
  code_cap_dict =  getMarketCapDict(all_code)
  sector_leader_dict = {}
  for key in thirdclass_code_dict.keys():
    code_list = thirdclass_code_dict[key]
    cap_list = [code_cap_dict[code] for code in code_list]
    cap_list = np.array(cap_list)
    leader_idx = np.argmax(cap_list)
    sector_leader_dict[key] = code_list[leader_idx]
    
  return sector_leader_dict


def A500Code():
  index_stock_cons_df = ak.index_stock_cons(symbol="000905")  # 中证A500的指数代码为000905
  A500_code_list = list(index_stock_cons_df['品种代码'])
  return A500_code_list

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
    df.rename(columns={
    '日期': 'Date',
    '股票代码': 'Code',
    '开盘': 'Open',
    '收盘': 'Close',
    '最高': 'High',
    '最低': 'Low',
    '成交量': 'Volume',
    '成交额': 'Amount',
    '振幅': 'Amplitude',
    '涨跌幅': 'ChangePct',
    '涨跌额': 'ChangeAmount',
    '换手率': 'TurnoverRate'
    },inplace=True)
    df_dict[code] = df
    # break
    
  with open(pickle_file, 'wb') as handle:
    pickle.dump(df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def LoadPickleData(pickle_path, verbose = False):
  if not os.path.exists(pickle_path):
    print("no file " + pickle_path)
    return
    
  with open(pickle_path, 'rb') as handle:
      df_dict = pickle.load(handle) 
      print(f"load pickle file: {pickle_path}")
  # if verbose:
    # print(f" {pickle_path}\n {df_dict}")
  
  return df_dict

def isTradeDay(trade_date=''):
  '''
  func: check if trade day
  args:
    trade_date: "%Y%m%d" string date, default is today
  '''
  if trade_date == '':
    trade_date = dt.date.today().strftime("%Y%m%d")
  df = ak.stock_zh_a_hist(symbol="000001", period = 'daily', start_date=trade_date, end_date= trade_date, adjust= 'qfq')
  if df.empty:
    return False
  else:
    return True

def DataIsUpdate(df_dict):
  '''
  func: check if data is update
  '''
  if df_dict is None:
    return False
  else:
    first_df = next(iter(df_dict.values()))
    last_date = first_df['Date'].iloc[-1]
    cur_date = dt.date.today().strftime("%Y%m%d")
    cur_df = ak.stock_zh_a_hist(symbol="000001", period = 'daily', start_date=last_date,
                                end_date= cur_date, adjust= 'qfq')
    if cur_df.empty:
      return True
    
    if cur_df['日期'].iloc[-1]== first_df['Date'].iloc[-1] and cur_df['收盘'].iloc[-1]== first_df['Close'].iloc[-1] and \
        cur_df['最低'].iloc[-1]== first_df['Low'].iloc[-1] and cur_df['最高'].iloc[-1]== first_df['High'].iloc[-1]:
       return True
    else:
      return False


def plot_pivots(X, pivots):
    plt.xlim(0, len(X))
    plt.ylim(X.min()*0.99, X.max()*1.01)
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    high_idx =[]
    low_idx =[]
    for key in  pivots.keys():
      if pivots[key] == 1:
        high_idx.append(key)
      else:
        low_idx.append(key)
    sorted(low_idx)
    sorted(high_idx)
      
    plt.scatter(high_idx, X[high_idx], color='r')
    plt.scatter(low_idx, X[low_idx], color='g')
    
    
def plot_pivot_line(X, pivots, enable_support = True, enable_resistance = True):
      ### fit low pivots
    # keep only -1 values
    if enable_support:
      low_pivots_index = [k for k, v in pivots.items() if v == -1]
      y = X[low_pivots_index]
      x = low_pivots_index

      data_cnt = len(X)
      data_range = range(0, data_cnt) 
      akima_interpolator = interpolate.Akima1DInterpolator(x, y)
      x_fit = np.linspace(min(data_range), max(data_range), data_cnt*2)
      y_fit = akima_interpolator(x_fit)
      plt.plot(x_fit, y_fit,'b')
    if enable_resistance:
      low_pivots_index = [k for k, v in pivots.items() if v == 1]
      y = X[low_pivots_index]
      x = low_pivots_index

      data_cnt = len(X)
      data_range = range(0, data_cnt) 
      akima_interpolator = interpolate.Akima1DInterpolator(x, y)
      x_fit = np.linspace(min(data_range), max(data_range), data_cnt*2)
      y_fit = akima_interpolator(x_fit)
      plt.plot(x_fit, y_fit,'r')

  
def show_stock_data_eastmoney(code, df_one, start_date="", end_date="", vline_data = [], save_dir = '', days = 100, predix = ''):
  '''
    vline_data:['2024-xx-xx']
  '''

  if start_date == "":
    start_date = dt.date.today() - dt.timedelta(days=100)
    start_date = start_date.strftime("%Y%m%d")

  if end_date == "":
    end_date = dt.date.today().strftime("%Y%m%d")
  df_one.reset_index(inplace=True)
  df_one['Date'] = pd.to_datetime(df_one['Date'])

  # 将Data列设置为索引，并转换为 datetime 类型
  df_one.set_index('Date', inplace=True)
  df_show = df_one.loc[start_date:end_date]

  # 定义 mplfinance 的自定义风格
  mc = mpf.make_marketcolors(up='r', down='g', volume='inherit')
  s = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc) # 

  # 使用 mplfinance 绘制 K 线图，并应用自定义风格
  fig_name = save_dir + predix+ code+".png"
  mpf.plot(df_show, type='candle', style=s,
       title=f"{predix} {code} K linechart",
       ylabel='Price',
       ylabel_lower='Vol',
       volume=True,
       vlines=dict(vlines=vline_data,linewidths=(1,)),
      #  mav=(5,20,250),
       show_nontrading=False,
       savefig=dict(fname=fig_name,dpi=100,pad_inches=0.25)
       )
  
  # mpf.show()
def outputFileInfo(df_dict):
  '''
  输出000001 最后1行数据
  '''
  first_df = next(iter(df_dict.values()))
  print(f"000001 info")
  print(first_df.tail(1))

def updateToLatestDay(pickle_file, period_unit, years):
  '''
  update data to latest day
  '''
  if not os.path.exists(pickle_file):
    dir = os.path.dirname(pickle_file)
    if not os.path.exists(dir):
      os.makedirs(dir)
    df = GetSecurityCode()  
    dump(df, pickle_file, period_unit , years)
    df_dict = LoadPickleData(pickle_file, True)
    return  df_dict
    
  df_dict = LoadPickleData(pickle_file, True)
  if DataIsUpdate(df_dict):
    print(f"data is latest, no need to update")
    print(df_dict["000402"].tail(1))
    return df_dict
  
  else:
    first_df = next(iter(df_dict.values()))
    last_date = first_df['Date'].iloc[-1]
    # last_date = last_date + dt.timedelta(days=1)
    cur_data = dt.date.today()
    last_date_str = last_date.strftime("%Y%m%d")
    cur_data_str = cur_data.strftime("%Y%m%d")   
    outputFileInfo(df_dict)
    print(f"df last day {last_date}, update data to {cur_data}")  
    for code, df in tqdm(df_dict.items()):
      # if  "000973" not in code:
      #   continue
      # print(f"code {code}")
      add_df = ak.stock_zh_a_hist(symbol=code, period = period_unit, start_date=last_date_str, end_date= cur_data_str, adjust= 'qfq')
      if  not add_df.empty:
        add_df.rename(columns={
          '日期': 'Date',
          '股票代码': 'Code',
          '开盘': 'Open',
          '收盘': 'Close',
          '最高': 'High',
          '最低': 'Low',
          '成交量': 'Volume',
          '成交额': 'Amount',
          '振幅': 'Amplitude',
          '涨跌幅': 'ChangePct',
          '涨跌额': 'ChangeAmount',
          '换手率': 'TurnoverRate'
          },inplace=True)
        if df['Date'].iloc[-1]== add_df['Date'].iloc[0]:
          df.drop(df.index[-1], inplace=True)
        df = pd.concat([df, add_df], ignore_index=True)
        df_dict[code] = df
      # break

    with open(pickle_file, 'wb') as handle:
      pickle.dump(df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
    return df_dict
     

if __name__ == '__main__':
  monthly_path = './sec_data/monthly.pickle' 
  weekly_path = './sec_data/weekly.pickle'
  daily_path = './sec_data/daily.pickle'

  updateToLatestDay(daily_path, 'daily', 1)
  # updateToLatestDay(weekly_path, 'weekly', 2)
  # updateToLatestDay(monthly_path, 'monthly', 1)
