import pandas as pd
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
import pickle
import mplfinance as mpf

  
def LoadAndShow(pickle_path):
  if not os.path.exists(pickle_path):
    print("no file " + pickle_path)
    return
    
  with open(pickle_path, 'rb') as handle:
      df_dict = pickle.load(handle) 
  
  print(df_dict["000001"])
  df = df_dict["000001"]

  get_stock_data_eastmoney(df)
  
  pass


def get_stock_data_eastmoney(df, start_date="", end_date=""):

  # 将日期列设置为索引，并转换为 datetime 类型
  df['日期'] = pd.to_datetime(df['日期'])
  df.set_index('日期', inplace=True)

  # 调整 DataFrame 列名以符合 mplfinance 的要求
  df.rename(columns={
    '开盘': 'Open',
    '收盘': 'Close',
    '最高': 'High',
    '最低': 'Low',
    '成交量': 'Volume'
  }, inplace=True)

  # 转换列名为小写，以符合 mplfinance 的要求
  df.columns = df.columns.str.lower()

  # 定义 mplfinance 的自定义风格
  mc = mpf.make_marketcolors(up='r', down='g', volume='inherit')
  s = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc, rc={'font.sans-serif': ['Microsoft YaHei']})
  code = '000001'
  # 使用 mplfinance 绘制 K 线图，并应用自定义风格
  mpf.plot(df, type='candle', style=s,
       title=f"{code} K 线图",
       ylabel='价格',
       ylabel_lower='成交量',
       volume=True,
       mav=(5,20,60),
       show_nontrading=False)

if __name__ == '__main__':
  pickle_path = '/home/yao/myproject/Stock/51_10天系列/01_数据操作/df.pickle' 
  LoadAndShow(pickle_path)
  