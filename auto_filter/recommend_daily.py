
import akshare as ak
import datetime as dt
import pandas as pd
import numpy as np
from  tqdm import tqdm
import os
import sys
import pickle
import mplfinance as mpf

import matplotlib.pyplot as plt

import data_utils
from tech.low_raiselimit_two import raiseLimitTwo


daily_pickle_path = "./sec_data/daily.pickle"
weekly_pickle_path = "./sec_data/weekly.pickle"
monthly_pickle_path = "./sec_data/monthly.pickle"


def dailyTechFilterAndPost():
  df_dict = data_utils.updateToLatestDay(daily_pickle_path, "daily")
  res = raiseLimitTwo(df_dict)
  print(f"res {res}")

  # sort res

  
  # save pic
  print(f"save pic")
  for code in tqdm(res):
    data_utils.show_stock_data_eastmoney(code, df_dict[code])


  return


#TODO: log
if __name__ == "__main__":
  # if not data_utils.isTradeDay():    
  #   print("Not trade day")
  #   exit()
   
  dailyTechFilterAndPost()
  # data_utils.updateToLatestDay(daily_pickle_path, "daily")
  # data_utils.updateToLatestDay(weekly_pickle_path, "weekly")
  # data_utils.updateToLatestDay(monthly_pickle_path, "monthly")

