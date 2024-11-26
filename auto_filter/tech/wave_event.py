'''
daily/weekly wave and exist horizon support, the longer the better
'''
import datetime as dt
import pandas as pd
import numpy as np
from  tqdm import tqdm
import os
import sys
import pickle
import mplfinance as mpf
import matplotlib.pyplot as plt
import cv2
from coordinate_geometry.point import *
from coordinate_geometry.equations import *

cur_file = os.path.abspath(__file__)
start_index = cur_file.find("auto_filter")
pro_path = cur_file[:start_index+11]
sys.path.append(pro_path)

import tech.tech_base as tech_base
import data_utils
import utils

cur_file = os.path.abspath(__file__)
start_index = cur_file.find("auto_filter")
pro_path = cur_file[:start_index+11]
sys.path.append(pro_path)

import tech.tech_base as tech_base
import data_utils

def fit_line(points):
    # 提取点的坐标
    x = points[:, 0]
    y = points[:, 1]
    
    # 计算均值
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # 计算斜率和截距
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    return slope, intercept

def point_to_line_distance(point, slope, intercept):
    # 计算点到直线的距离
    x, y = point
    distance = abs(slope * x - y + intercept) / np.sqrt(slope ** 2 + 1)
    return distance
def findFoot(a, b, c, x1, y1): 

  temp = (-1 * (a * x1 + b * y1 + c) // (a * a + b * b)) 
  x = temp * a + x1 
  y = temp * b + y1 
  return (x, y) 


def waveSupportHorImpl(df_one, pivots, last_price_ratio,  hor_diff_ratio, high_pivot_price_diff, show = False):
  '''
  '''
  price_1 = np.array(df_one["Close"])
  low_pivots_index = [k for k, v in pivots.items() if v == -1]
  high_pivots_index = [k for k, v in pivots.items() if v == 1]
  if (len(low_pivots_index) < 3):
    return False, 0, 0,0
  if (len(high_pivots_index) < 2):
    return False, 0, 0,0
  
  # 波谷在波峰前面，否则退出
  if (high_pivots_index[-1] > low_pivots_index[-1]):
    return False, 0, 0,0
  
  last_price = np.asarray(price_1)[-1]
  last_lowest_pivot_price = np.asarray(price_1)[low_pivots_index[-1]]
  if last_price > last_price_ratio * last_lowest_pivot_price :
    return False, 0, 0,0
  
  
  # get support length
  thirdlatest_low_pivot_index = low_pivots_index[-3]
  secondlatest_low_pivot_index = low_pivots_index[-2]
  latest_low_pivot_index = low_pivots_index[-1]
  thirdlatest_low_pivot_price = np.asarray(price_1)[thirdlatest_low_pivot_index]
  secondlatest_low_pivot_price = np.asarray(price_1)[secondlatest_low_pivot_index]
  latest_low_pivot_price = np.asarray(price_1)[latest_low_pivot_index]
  if (thirdlatest_low_pivot_price - secondlatest_low_pivot_price) / thirdlatest_low_pivot_price > hor_diff_ratio or \
     (secondlatest_low_pivot_price - latest_low_pivot_price) / secondlatest_low_pivot_price > hor_diff_ratio:
    return False, 0, 0,0
  
  #hor line
  support_price_list = np.array([thirdlatest_low_pivot_price, secondlatest_low_pivot_price, latest_low_pivot_price])
  mean_support_price = np.mean(support_price_list)
  diff_sum = abs(thirdlatest_low_pivot_price-mean_support_price)/ thirdlatest_low_pivot_price + \
             abs(secondlatest_low_pivot_price-mean_support_price)/ secondlatest_low_pivot_price + \
             abs(latest_low_pivot_price-mean_support_price)/ latest_low_pivot_price

  # two wave ratio
  secondlatest_high_pivot_index = high_pivots_index[-2]
  latest_high_pivot_index = high_pivots_index[-1]
  secondlatest_high_pivot_price = np.asarray(price_1)[secondlatest_high_pivot_index]
  latest_high_pivot_price = np.asarray(price_1)[latest_high_pivot_index]
  if abs(secondlatest_high_pivot_price - latest_high_pivot_price) / secondlatest_high_pivot_price > high_pivot_price_diff:
    return False, 0, 0,0
  
  wave_raise_mean = (secondlatest_high_pivot_price/mean_support_price + latest_high_pivot_price/mean_support_price) / 2
  
  return  True, diff_sum, wave_raise_mean, mean_support_price
  
def waveSupportSlopeImpl(df_one, pivots,last_price_ratio, slope_ratio, high_pivot_price_diff ,show = False):
  '''
  '''
  price_1 = np.array(df_one["Close"])
  low_pivots_index = [k for k, v in pivots.items() if v == -1]
  high_pivots_index = [k for k, v in pivots.items() if v == 1]
  if (len(low_pivots_index) < 3):
    return False, 0, 0,0
  if (len(high_pivots_index) < 2):
    return False, 0, 0,0
  
  # 波谷在波峰前面，否则退出
  if (high_pivots_index[-1] > low_pivots_index[-1]):
    return False, 0, 0,0
  
  last_price = np.asarray(price_1)[-1]
  last_lowest_pivot_price = np.asarray(price_1)[low_pivots_index[-1]]
  if last_price > last_price_ratio * last_lowest_pivot_price :
    return False, 0, 0,0
  
  
  # get support length
  thirdlatest_low_pivot_index = low_pivots_index[-3]
  secondlatest_low_pivot_index = low_pivots_index[-2]
  latest_low_pivot_index = low_pivots_index[-1]
  thirdlatest_low_pivot_price = np.asarray(price_1)[thirdlatest_low_pivot_index]
  secondlatest_low_pivot_price = np.asarray(price_1)[secondlatest_low_pivot_index]
  latest_low_pivot_price = np.asarray(price_1)[latest_low_pivot_index]
  if (thirdlatest_low_pivot_price > secondlatest_low_pivot_price)  or \
     (secondlatest_low_pivot_price > latest_low_pivot_price):
    return False, 0, 0,0
  if abs(thirdlatest_low_pivot_price - latest_low_pivot_price) / thirdlatest_low_pivot_price > slope_ratio:
    return False, 0, 0,0

  #slope line, fit
  support_price_pt = np.array([[thirdlatest_low_pivot_index,thirdlatest_low_pivot_price],\
                              [secondlatest_low_pivot_index, secondlatest_low_pivot_price], \
                              [latest_low_pivot_index, latest_low_pivot_price]])
  
  slope, intercept = fit_line(support_price_pt)
  test_point3 = np.array([thirdlatest_low_pivot_index,thirdlatest_low_pivot_price])
  distance3 = point_to_line_distance(test_point3, slope, intercept)
  diff3 = abs(distance3) / thirdlatest_low_pivot_price

  test_point2 = np.array([secondlatest_low_pivot_index, secondlatest_low_pivot_price])
  distance2 = point_to_line_distance(test_point2, slope, intercept)
  diff2 = abs(distance2) / secondlatest_low_pivot_price

  test_point1 = np.array([latest_low_pivot_index, latest_low_pivot_price])
  distance1 = point_to_line_distance(test_point1, slope, intercept)
  diff1 = abs(distance1) / latest_low_pivot_price

  diff_sum = diff3 + diff2 + diff1
  
  # high pivot price
  secondlatest_high_pivot_index = high_pivots_index[-2]
  latest_high_pivot_index = high_pivots_index[-1]
  secondlatest_high_pivot_price = np.asarray(price_1)[secondlatest_high_pivot_index]
  latest_high_pivot_price = np.asarray(price_1)[latest_high_pivot_index]
  
  if abs(secondlatest_high_pivot_price - latest_high_pivot_price) / secondlatest_high_pivot_price > high_pivot_price_diff:
    return False, 0, 0,0
  h_pt1 = np.array([secondlatest_high_pivot_index, secondlatest_high_pivot_price])
  h_pt2 = np.array([latest_high_pivot_index, latest_high_pivot_price])
  
  support_line = equation_type1(slope, intercept)
  dist_price1 = support_line.distance(point(h_pt1[0], h_pt1[1]))
  dist_price2 = support_line.distance(point(h_pt2[0], h_pt2[1]))
  # ax + by + c = 0 
  foot_pt1= support_line.foot_of_perpendicular(point(h_pt1[0], h_pt1[1]))
  foot_pt2 = support_line.foot_of_perpendicular(point(h_pt2[0], h_pt2[1]))
  
  raise_mean = (dist_price1/foot_pt1.y + dist_price2/foot_pt2.y) / 2
  
  # plt.plot(price_1)
  # x1 = 0
  # y1 = x1 * slope + intercept
  # x2 = len(price_1)
  # y2 = x2 * slope + intercept
  # plt.plot([x1, x2], [y1, y2], color='r')  
  # plt.scatter([h_pt1[0], h_pt2[0]], [h_pt1[1], h_pt2[1]], color='b')
  # plt.scatter([foot_pt1.x, foot_pt2.x], [foot_pt1.y, foot_pt2.y], color='r')
  # plt.savefig("test.png")
  # plt.show()

  return True, diff_sum, raise_mean, \
    np.array([h_pt1, h_pt2, np.asarray([foot_pt1.x, foot_pt1.y]), np.asarray([foot_pt2.x, foot_pt2.y]), np.asarray([slope, intercept])])

param_config = {
  "high_pivot_price_diff":
  {
    "daily": 0.08,
    "weekly": 0.15
  },
  "last_price_ratio":
  {
    "daily": 1.04,
    "weekly": 1.06
  },
  "hor_diff_ratio": {
    "daily": 0.03,
    "weekly": 0.05
  },
  "slope_ratio": {
    "daily": 0.14,
    "weekly": 0.2
  },
  "daily": {
    "raise_ratio": 0.103,
    "decade_ratio": 0.09
  },
  "weekly": {
    "raise_ratio": 0.15,
    "decade_ratio": 0.11
  }
}
def GetWaveSupportDaily(df_dict, order_cnt = 15, show = False):
  slope_dict = {}
  horizon_dict = {}
  slope_support = {}
  hor_support_price_dict = {}

  print("daily wave support")
  for code, value in tqdm(df_dict.items()):
    if value["Close"].iloc[-1] > 50:
      continue
    pivots = tech_base.get_pivots(value["Close"], param_config["daily"]["raise_ratio"], param_config["daily"]["decade_ratio"])
    sel_h, diff_ratio, mean_raise, support_price = waveSupportHorImpl(value, pivots, param_config["last_price_ratio"]["daily"], 
                                                       param_config["hor_diff_ratio"]["daily"], param_config["high_pivot_price_diff"]["daily"] ,show=show)
    sel_s, s_ratio, slope_mean_raise, support_line = waveSupportSlopeImpl(value, pivots, param_config["last_price_ratio"]["daily"], 
                                                      param_config["slope_ratio"]["daily"],param_config["high_pivot_price_diff"]["daily"], show=show)
    if sel_h and diff_ratio < 0.05:
      horizon_dict[code] = mean_raise
      hor_support_price_dict[code] = support_price
    if sel_s and s_ratio < 0.05:
      slope_dict[code] = slope_mean_raise
      slope_support[code] = support_line
    if show:
      data_utils.plot_pivots(value["Close"], pivots)
      data_utils.plot_pivot_line(value["Close"], pivots)
      plt.show()

  hor_sort_dict =  dict(sorted(horizon_dict.items(), key=lambda x: x[1], reverse=True))
  hor_sort_dict_cnt = dict(list(hor_sort_dict.items())[:order_cnt])

  slope_sort_dict =  dict(sorted(slope_dict.items(), key=lambda x: x[1], reverse=True))
  slope_sort_dict_cnt = dict(list(slope_sort_dict.items())[:order_cnt])
  
  return hor_sort_dict_cnt, hor_support_price_dict, slope_sort_dict_cnt, slope_support

def GetWaveSupportWeekly(df_dict, order_cnt = 15, show = False):
  slope_dict = {}
  horizon_dict = {}
  slope_support = {}
  hor_support_price_dict = {}

  for code, value in df_dict.items():
    if value["Close"].iloc[-1] > 50:
      continue

    pivots = tech_base.get_pivots(value["Close"],  param_config["weekly"]["raise_ratio"], param_config["weekly"]["decade_ratio"])
    sel_h, diff_ratio, mean_raise, support_price = waveSupportHorImpl(value, pivots, param_config["last_price_ratio"]["weekly"], param_config["hor_diff_ratio"]["weekly"],\
                                                                      param_config["high_pivot_price_diff"]["weekly"], show=show)
    sel_s, s_ratio, slope_mean_raise, support_line = waveSupportSlopeImpl(value, pivots, param_config["last_price_ratio"]["weekly"], param_config["slope_ratio"]["weekly"],\
                                                                      param_config["high_pivot_price_diff"]["weekly"], show=show)
    
    if sel_h and diff_ratio < 0.08:
      horizon_dict[code] = mean_raise
      hor_support_price_dict[code] = support_price
    if sel_s and s_ratio < 0.08:
      slope_dict[code] = slope_mean_raise
      slope_support[code] = support_line
    if show:
      data_utils.plot_pivots(value["Close"], pivots)
      data_utils.plot_pivot_line(value["Close"], pivots)
      plt.show()

  hor_sort_dict =  dict(sorted(horizon_dict.items(), key=lambda x: x[1], reverse=True))
  hor_sort_dict_cnt = dict(list(hor_sort_dict.items())[:order_cnt])

  slope_sort_dict =  dict(sorted(slope_dict.items(), key=lambda x: x[1], reverse=True))
  slope_sort_dict_cnt = dict(list(slope_sort_dict.items())[:order_cnt])
  
  return hor_sort_dict_cnt, hor_support_price_dict, slope_sort_dict_cnt, slope_support


test_map = {
  # "600855": ["20240911"],  #zhong chong gufen
  # "001965": ["20241127"],  
  "601928": ["20241127"],  
  # "603050": ["20240126"],  # 缠+突破

}

def test():
  pass

def plotHorSupport(code, df_one, support_price_dict, save_dir, time_period,cnt):
  '''
  time_period: "daily", "weekly"
  '''
  price = df_one["Close"]
  pivots = tech_base.get_pivots(price, param_config[time_period]["raise_ratio"], param_config[time_period]["decade_ratio"])
  data_utils.plot_pivots(price, pivots)
  plt.axhline(support_price_dict[code])
  # plt.show()
  # plt.savefig(save_dir+f"{time_period}_hor_{cnt:03}_{code}.png")
  # plt.clf()
  
def plotSlopeSupport(code, df_one, slope_support, save_dir, time_period, cnt):
  '''
  time_period: "daily", "weekly"
  '''
  price = df_one["Close"]
  pivots = tech_base.get_pivots(price, param_config[time_period]["raise_ratio"], param_config[time_period]["decade_ratio"])
  data_utils.plot_pivots(price, pivots)
  cur_support = slope_support[code]
  x1 = 0
  y1 = x1 * cur_support[-1][0] + cur_support[-1][1]
  x2 = len(price)
  y2 = x2 * cur_support[-1][0] + cur_support[-1][1]
  plt.plot([x1, x2], [y1, y2], color='r')  
  # plt.show()
  plt.savefig(save_dir+f"{time_period}_slope_{cnt:03}_{code}.png")
  plt.clf()

if __name__ == '__main__':
  df_dict_daily = data_utils.LoadPickleData(pro_path+"/sec_data/daily.pickle")
  df_dict_weekly = data_utils.LoadPickleData(pro_path+"/sec_data/weekly.pickle")
  test_cnt = 0
  for key, val in test_map.items():
    test_cnt += val.__len__()

  test_dict = {}
  for key, val in test_map.items():
    test_dict[key] = df_dict_daily[key]
    for date in val:
      end_day = dt.datetime.date(dt.datetime.strptime(date, "%Y%m%d"))
      test_dict[key] = test_dict[key][test_dict[key]["Date"] <= end_day]
    
  # test_dict = df_dict_weekly
  test_dict = df_dict_daily
  hor20m, support_price_dict, slope20, slope_support = GetWaveSupportDaily(test_dict, show=False)
  whor20m, wsupport_price_dict, wslope20, wslope_support = GetWaveSupportWeekly(test_dict, show=False)
  save_dir = utils.getProjectPath("auto_filter")+ "/workdata/"

  for i, (code, ss) in tqdm(enumerate(hor20m.items())):
    plotHorSupport(code, test_dict[code], support_price_dict, save_dir, "daily",i)
    
  for i, (code, ss) in tqdm(enumerate(slope20.items())):
    plotSlopeSupport(code, test_dict[code], slope_support, save_dir, "daily",i)
    
    
  for i, (code, ss) in tqdm(enumerate(whor20m.items())):
    plotHorSupport(code, test_dict[code], wsupport_price_dict, save_dir, "weekly",i)
  for i, (code, ss) in tqdm(enumerate(wslope20.items())):
    plotSlopeSupport(code, test_dict[code], wslope_support, save_dir, "weekly",i)
    
  print(f"daily_hor {hor20m}\n")
  print(f"daily_slope {slope20}")
  print(f"weekly_hor {whor20m}\n")
  print(f"weekly_slope {wslope20}")
