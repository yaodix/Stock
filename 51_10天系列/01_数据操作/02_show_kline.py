#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import pandas as pd
import numpy as np
import os
import datetime as dt

import cv2
import matplotlib
import matplotlib.pyplot as plt
import pickle
import mplfinance as mpf
import tqdm as tqdm

plt.ion()

def LoadData(pickle_path):
  if not os.path.exists(pickle_path):
    print("no file " + pickle_path)
    return
    
  with open(pickle_path, 'rb') as handle:
      df_dict = pickle.load(handle) 
  
  return df_dict

def show_stock_data_eastmoney(code, df_one, start_date="20200630", end_date="20240530"):
  # 将Data列设置为索引，并转换为 datetime 类型

  df_one['Date'] = pd.to_datetime(df_one['Date'])
  df_one.set_index('Date', inplace=True)
  df_one = df_one.loc[start_date:end_date]

  # 调整 DataFrame 列名以符合 mplfinance 的要求
  df_show = df_one.rename(columns={
    '开盘': 'Open',
    '收盘': 'Close',
    '最高': 'High',
    '最低': 'Low',
    '成交量': 'Volume'
  })

  # 转换列名为小写，以符合 mplfinance 的要求
  df_show.columns = df_show.columns.str.lower()

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
      #  mav=(5,20,250),
       show_nontrading=False,
      #  savefig=dict(fname=fig_name,dpi=100,pad_inches=0.25)
       )
  # mpf.show()

   
if __name__ == '__main__':
  pickle_path = '/home/yao/workspace/Stock/51_10天系列/01_数据操作/df_0707.pickle' 
  df_dict = LoadData(pickle_path)
  
  for key, val in tqdm.tqdm(df_dict.items()):
    # filter
    # if val.iloc[-1]['收盘'] / val.iloc[-30]['收盘'] < 1.1:
      # continue

    end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
    end_day = end_day.strftime("%Y%m%d")   

    show_stock_data_eastmoney(key, val, "20240101", end_day)
    plt.draw()
    plt.pause(0.5)   # 无pause无法显示图像
    user_input = input("按任意键继续，输入'q'退出: ")
    plt.close('all')
    
    
