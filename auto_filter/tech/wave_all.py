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

cur_file = os.path.abspath(__file__)
start_index = cur_file.find("auto_filter")
pro_path = cur_file[:start_index+11]
sys.path.append(pro_path)

import tech.tech_base as tech_base
import data_utils

def angelThreePoint(p_center, p1, p2):
  ba = p1 - p_center
  bc = p2 - p_center

  cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
  angle = np.arccos(cosine_angle)

  return np.degrees(angle)


def removeReverseGap(wave_m):
  del_pts_idx2 = []
  for i in range(0, len(wave_m)-3):
    if wave_m[i+3][1] > wave_m[i+1][1] and wave_m[i+3][1] > wave_m[i+2][1] and wave_m[i+1][1] > wave_m[i][1] and \
       wave_m[i+2][1] < wave_m[i+1][1] and \
       wave_m[i+2][1] > wave_m[i][1]+abs(wave_m[i+1][1]-wave_m[i][1])*0.3:
      if abs(wave_m[i+2][1]-wave_m[i+1][1])/wave_m[i+1][1] < 0.05:
        del_pts_idx2.append(i+1)
        del_pts_idx2.append(i+2)
    elif wave_m[i+3][1] < wave_m[i+1][1] and wave_m[i+3][1] < wave_m[i+2][1] and wave_m[i+1][1] < wave_m[i][1] and \
         wave_m[i+2][1] > wave_m[i+1][1] and \
        wave_m[i+2][1] < wave_m[i][1]-abs(wave_m[i][1]-wave_m[i+1][1])*0.3:
      if abs(wave_m[i+2][1]-wave_m[i+1][1])/wave_m[i+1][1] < 0.05:
        del_pts_idx2.append(i+1)
        del_pts_idx2.append(i+2)
    else:
      continue
  wave_m2 = np.delete(wave_m, np.array(del_pts_idx2), axis=0)
  return wave_m2

def removeForwardGap(wave_in):
  del_pts_idx = []
  for i in range(0, len(wave_in)-2):
    if wave_in[i+2][1] >= wave_in[i+1][1] and wave_in[i+1][1] > wave_in[i][1]:
      del_pts_idx.append(i+1)
    elif wave_in[i+2][1] <= wave_in[i+1][1] and wave_in[i+1][1] < wave_in[i][1]:
      del_pts_idx.append(i+1)
    else:
      continue
  wave_m = np.delete(wave_in, np.array(del_pts_idx), axis=0)
  return wave_m

def removeMidV(wave_in):
  all_v_list = []
  code_list = []
  for i in range(1, len(wave_in)-1):
    if wave_in[i][1] < wave_in[i+1][1] and wave_in[i][1] < wave_in[i-1][1]:
      all_v_list.append(wave_in[i])
      code_list.append(i)
      
  del_idx_list = []
  for i in range(1, len(all_v_list)-1):
    if all_v_list[i][1] > (wave_in[code_list[i]-1][1]- all_v_list[i-1][1])*0.3+ all_v_list[i-1][1] and \
      all_v_list[i][1] > (wave_in[code_list[i]+1][1]- all_v_list[i+1][1])*0.3+ all_v_list[i+1][1]:
      del_idx_list.append(code_list[i])
  
      
  wave_m = np.delete(wave_in, np.array(del_idx_list), axis=0)
  return wave_m

def mergeWave(wave_in):
  '''
  '''
  # 同趋势的点，删除中间点
  wave_m = removeForwardGap(wave_in)
  # plt.scatter(wave_m[:, 0], wave_m[:, 1], c='b', s=5)
  # plt.show()

  # 同趋势的点，删除中间小逆势点
  # wave_m2 = removeReverseGap(wave_m)
  wave_m3 = removeMidV(wave_m)
  # wave_m4 = removeForwardGap(wave_m3)
  
  
  return wave_m3
    
  
def getAllWave(df_one, show = False):
  '''
  '''
  include_cnt = 180
  price_1 = df_one["Close"][-include_cnt:]
  idx = np.arange(len(price_1))
  points = np.column_stack((idx, price_1)).astype(np.float32)
  poly = cv2.approxPolyDP((points), 0.06, False)
  poly = poly.reshape(-1, 2)
  
  merge_pts = mergeWave(poly)
  
  plt.plot(np.arange(len(price_1)), price_1, 'k:', alpha=0.5)
  # plt.scatter(poly[:, 0], poly[:, 1], c='b', s=10)
  plt.scatter(merge_pts[:, 0], merge_pts[:, 1], c='b', s=5)
  plt.show()
  
  
  pass

def waveSupportImpl(df_one, raise_ratio, decade_ratio, straight_angle_thresh = 165, show = False):
  '''
  '''
  price_1 = np.array(df_one["Close"])[-180:]
  dateProcess = np.array(df_one["Date"])[-180:]
  pivots = tech_base.get_pivots(price_1, 0.103, 0.09)
  if show:
    data_utils.plot_pivots(price_1, pivots)
    data_utils.plot_pivot_line(price_1, pivots)
    plt.show()
  low_pivots_index = [k for k, v in pivots.items() if v == -1]
  high_pivots_index = [k for k, v in pivots.items() if v == 1]
  if (len(low_pivots_index) < 3):
    return 0, 0, 0, 0
  if (len(high_pivots_index) < 2):
    return 0, 0, 0, 0
  
  # 波谷在波峰前面，否则退出
  if (high_pivots_index[-1] > low_pivots_index[-1]):
    return 0, 0, 0, 0
  
  last_price = np.asarray(price_1)[-1]
  last_lowest_pivot_price = np.asarray(price_1)[low_pivots_index[-1]]
  if last_price > 1.05 * last_lowest_pivot_price :
    return 0, 0, 0, 0
  
  
  # get support length
  valid_suport_start_idx = -1
  pivots_cnt = 1
  for i in range(len(low_pivots_index)-2, 0, -1):
    angle = angelThreePoint(np.array([i,   price_1[low_pivots_index[i]]]),
                            np.array([i+1, price_1[low_pivots_index[i+1]]]),
                            np.array([i-1, price_1[low_pivots_index[i-1]]]))
    if angle > straight_angle_thresh:
      valid_suport_start_idx = i-1
      pivots_cnt +=1
    else:
      break  
  
  start_date = dateProcess[low_pivots_index[valid_suport_start_idx]]
  mid_date = dateProcess[low_pivots_index[int((valid_suport_start_idx+len(low_pivots_index))/2)]]
  
  return len(price_1) - low_pivots_index[valid_suport_start_idx], start_date, mid_date, pivots_cnt
  
def angleRefHorizonRatio(code_value, start_date, mid_date):
    start_index = [i for i, date in enumerate(code_value["Date"]) if date == start_date]
    mid_index = [i for i, date in enumerate(code_value["Date"]) if date == mid_date]
    price_start = code_value['Close'].iloc[start_index[0]]
    price_mid = code_value['Close'].iloc[mid_index[0]]
    price_end = code_value['Close'].iloc[-1]
    
    angle_end = angelThreePoint(np.array([start_index[0], price_start]),
                            np.array([len(code_value["Date"]), price_start]),
                            np.array([len(code_value["Date"]), price_end]))
    angle_mid = angelThreePoint(np.array([start_index[0], price_start]),
                            np.array([mid_index[0], price_start]),
                            np.array([mid_index[0], price_mid]))
    price_ratio_start = abs(price_end - price_start) / price_start
    price_ratio_mid = abs(price_end - price_mid) / price_mid
    
    return max(angle_end, angle_mid), max(price_ratio_start, price_ratio_mid)

param_config = {
  "daily": {
    "raise_ratio": 0.103,
    "decade_ratio": 0.09
  },
  "weekly": {
    "raise_ratio": 0.15,
    "decade_ratio": 0.11
  }
}
def GetWaveSupportDaily(df_dict, order_cnt = 10, show = False):
  slope_dict = {}
  horizon_dict = {}
  pivots_cnt_dict = {}
  for code, value in df_dict.items():
    len, start_date, mid_date, pivots_cnt = waveSupportImpl(value, param_config["daily"]["raise_ratio"], param_config["daily"]["decade_ratio"],show=show)
    if len > 10:
      angle, ratio= angleRefHorizonRatio(value, start_date, mid_date)
      if angle < 5 and ratio < 0.1:
        horizon_dict[code] = len
        pivots_cnt_dict[code] = pivots_cnt
      else:
        slope_dict[code] = len      
        pivots_cnt_dict[code] = pivots_cnt
  
  hor_sort_dict =  dict(sorted(horizon_dict.items(), key=lambda x: x[1], reverse=True))
  hor_sort_dict_20 = dict(list(hor_sort_dict.items())[:order_cnt])
  hor_sort_dict_20 = dict(sorted(hor_sort_dict_20.items(), key=lambda x: pivots_cnt_dict[x[0]]))
  
  slope_sort_dict =  dict(sorted(slope_dict.items(), key=lambda x: x[1], reverse=True))
  slope_sort_dict_20 = dict(list(slope_sort_dict.items())[:order_cnt])
  slope_sort_dict_20 = dict(sorted(slope_sort_dict_20.items(), key=lambda x: pivots_cnt_dict[x[0]]))
  
  
  return hor_sort_dict_20, slope_sort_dict_20

def GetWaveSupportWeekly(df_dict, order_cnt = 10, show = False):
  slope_dict = {}
  horizon_dict = {}
  pivots_cnt_dict = {}
  for code, value in df_dict.items():
    len, start_date, mid_date, pivots_cnt = waveSupportImpl(value, param_config["weekly"]["raise_ratio"], param_config["weekly"]["decade_ratio"],show=show)
    if len > 10:
      angle, ratio= angleRefHorizonRatio(value, start_date, mid_date)
      if angle < 5 and ratio < 0.1:
        horizon_dict[code] = len
        pivots_cnt_dict[code] = pivots_cnt
      else:
        slope_dict[code] = len      
        pivots_cnt_dict[code] = pivots_cnt
  
  hor_sort_dict =  dict(sorted(horizon_dict.items(), key=lambda x: x[1], reverse=True))
  hor_sort_dict_20 = dict(list(hor_sort_dict.items())[:order_cnt])
  hor_sort_dict_20 = dict(sorted(hor_sort_dict_20.items(), key=lambda x: pivots_cnt_dict[x[0]]))
  
  slope_sort_dict =  dict(sorted(slope_dict.items(), key=lambda x: x[1], reverse=True))
  slope_sort_dict_20 = dict(list(slope_sort_dict.items())[:order_cnt])
  slope_sort_dict_20 = dict(sorted(slope_sort_dict_20.items(), key=lambda x: pivots_cnt_dict[x[0]]))
  
  
  return hor_sort_dict_20, slope_sort_dict_20


test_map = {
  # "600855": ["20240911"],  #zhong chong gufen
  # "001965": ["20241127"],  
  "601928": ["20241127"],  
  # "603050": ["20240126"],  # 缠+突破

}

def test():
  pass


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
  # hor20m, slope20 = GetWaveSupportDaily(test_dict, show=True)
  # hor20m, slope20 = GetWaveSupportWeekly(test_dict, show=False)
  getAllWave(test_dict["603444"],)
  # print(f"hor {hor20m}\n")
  # print(f"slope {slope20}")
