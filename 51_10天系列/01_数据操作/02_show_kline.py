#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import pandas as pd
import numpy as np
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import pickle
import mplfinance as mpf
import tqdm as tqdm

# 设置 Matplotlib 支持中文显示
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 设置中文字体，确保中文能够正常显示
# plt.rcParams['axes.unicode_minus'] = False # 解决负号 '-' 显示为方块的问题

plt.ion()

def LoadData(pickle_path):
  if not os.path.exists(pickle_path):
    print("no file " + pickle_path)
    return
    
  with open(pickle_path, 'rb') as handle:
      df_dict = pickle.load(handle) 
  
  return df_dict


def get_stock_data_eastmoney(df_one, start_date="20200630", end_date="20240530"):
  # 将日期列设置为索引，并转换为 datetime 类型
  df_one['日期'] = pd.to_datetime(df_one['日期'])
  df_one.set_index('日期', inplace=True)
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
  # s = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc, rc={'font.sans-serif': ['Microsoft YaHei']}) # 
  s = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc) # 

  code = df_show["股票代码"][0]

  # 使用 mplfinance 绘制 K 线图，并应用自定义风格
  mpf.plot(df_show, type='candle', style=s,
       title=f"{code} K linechart",
       ylabel='Price',
       ylabel_lower='Vol',
       volume=True,
       mav=(5,20,60,250),
       show_nontrading=False)
   
if __name__ == '__main__':
  pickle_path = '/home/yao/myproject/Stock/51_10天系列/01_数据操作/df.pickle' 
  df_dict = LoadData(pickle_path)
  
  for key, val in tqdm.tqdm(df_dict.items()):
    get_stock_data_eastmoney(val, "20240101", "20240530")
    input()
    plt.close('all')
    
    
