# SwClass loader

import json
import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import re
cur_file = os.path.abspath(__file__)
start_index = cur_file.find("auto_filter")
pro_path = cur_file[:start_index+11]
sys.path.append(pro_path)

import tech.tech_base as tech_base
import data_utils

def sectorRaiseSort(df_dict, class_dict, days ):
  '''
  sort raise of n days
  args: 
    class_dict: dict, key: class, value: list of code
    days: int, days to raise
    number_include: int, number of code to include
  '''
  class_raise_dict_val = {}
  class_raise_dict_code = {}
  for key in class_dict.keys():
    code_list = class_dict[key]
    valid_code_raise_dict = {}
    for code in code_list:
      if code not in df_dict.keys():
        continue
      df = df_dict[code]
      raise_ratio = 0
      for i in range(days, 0, -1):
        raise_ratio += df["ChangePct"].iloc[-i] 
      valid_code_raise_dict[code] = float(raise_ratio)
     
    sorted_raise =  dict(sorted(valid_code_raise_dict.items(), key=lambda x: x[1], reverse=True))
    include_cnt = max(3, int(sorted_raise.__len__()/4))
    raise_big = dict(list(sorted_raise.items())[:include_cnt])
    class_raise_dict_val[key] = np.array(list(raise_big.values())).mean()
    class_raise_dict_code[key] = list(raise_big.keys())
    
  return dict(sorted(class_raise_dict_val.items(), key=lambda x: x[1], reverse=True)), class_raise_dict_code
    
  
def sectorTech(df_dict):
  class_dict, _ , _=  data_utils.LoadSwClassDict()
  sorted_class_dict_val, sorted_class_dict_code = sectorRaiseSort(df_dict, class_dict, 1)
  first_5_dict = dict(list(sorted_class_dict_val.items())[:5])
  first_5_dict_code = [sorted_class_dict_code[key] for key in first_5_dict]  
  for i, (key, value) in enumerate(first_5_dict.items()):
    print(f"{key}: {value} , {first_5_dict_code[i]}")

  
  
def sectorRotation(df_dict, class_dict, days, number_include):
  '''
  sort raise of n days
  args:
    class_dict: dict, key: class, value: list of code
    days: int, days to raise
    number_include: int, number of code to include
  '''
  pass

def test(df_dict):
  end_day = dt.date(dt.date.today().year,dt.date.today().month,dt.date.today().day)
  test_dict = {}
  for i in range(6, -1, -1):
    test_day = end_day - dt.timedelta(days=i)
    test_day_str = test_day.strftime("%Y%m%d")
    if not data_utils.isTradeDay(test_day_str):
      continue

    for key in df_dict.keys():
          test_dict[key] = df_dict[key][df_dict[key]["Date"] <= test_day]
          
    print(f"-----------------------test_day end {test_day}")
    print(test_dict["000001"].tail(1))
    sectorTech(test_dict)


if __name__ == '__main__':
  df_dict = data_utils.LoadPickleData(pro_path+"/sec_data/daily.pickle")
  