
import akshare as ak
import datetime as dt
import pandas as pd
import numpy as np
from  tqdm import tqdm
import os
import sys
import pickle
import mplfinance as mpf
import yagmail
import utils

import matplotlib.pyplot as plt

import data_utils
from tech.low_raiselimit_two import raiseLimitTwo
from tech.wave_raise import waveTechFilter
from tech.wave_struct import GetWaveStructureWeekly
from tech.wave_event import GetWaveSupportDaily, GetWaveSupportWeekly, plotHorSupport, plotSlopeSupport
import warnings
warnings.filterwarnings("ignore")

def commonKeys(dict1, dict2):
  return list(set(dict1.keys()) & set(dict2.keys()))

yag = yagmail.SMTP( user="dapaier1115@163.com", password="AXCKERNKNVWSNLVJ", host='smtp.163.com')
mail_send_list = ["liuyao199111@163.com"]


daily_pickle_path = "./sec_data/daily.pickle"
weekly_pickle_path = "./sec_data/weekly.pickle"
monthly_pickle_path = "./sec_data/monthly.pickle"


def dailyTechFilterAndPost():
  df_dict = data_utils.updateToLatestDay(daily_pickle_path, "daily", 1)
  # code_dict = raiseLimitTwo(df_dict)
  # print(f"raiseLimitTwo size {code_dict.__len__()} {code_dict.keys()}")

  # wave_low_dcit, wave_high_dict = waveTechFilter(df_dict,enable_high=True)
  # print(f"wave_low_dcit size {wave_low_dcit.__len__()} {wave_low_dcit.keys()}")
  # print(f"wave_high_dict {wave_high_dict.__len__()} {wave_high_dict.keys()}")
  daily_horizon_dict, support_price_dict, daily_slope_dict, slope_support = GetWaveSupportDaily(df_dict, show=False)

  print(f"daily_horizon_dict {daily_horizon_dict.__len__()} {daily_horizon_dict.keys()}")
  print(f"daily_slope_dict {daily_slope_dict.__len__()} {daily_slope_dict.keys()}")

  weekly_df_dict = data_utils.updateToLatestDay(weekly_pickle_path, "weekly", 5)
  weekly_horizon_dict, wsupport_price_dict, weekly_slope_dict, wslope_support = GetWaveSupportWeekly(weekly_df_dict, show=False)
  print(f"weekly_horizon_dict {weekly_horizon_dict.__len__()} {weekly_horizon_dict}")
  print(f"weekly_slope_dict {weekly_slope_dict.__len__()} {weekly_slope_dict}")
  # save pic
  print(f"save pic")
  save_dir = utils.getProjectPath("auto_filter")+ "/workdata/"
  for file in os.listdir(save_dir):
    if file.endswith('.png'):
      os.remove(save_dir+file)

  mail_cont = ["tech!",]
  # for code, data_idx in tqdm(code_dict.items()):
  #   data_utils.show_stock_data_eastmoney(code, df_dict[code], save_dir= save_dir, predix="tr_")
  #   fig_name = save_dir+"tr_" + code+".png"
  #   mail_cont.append(yagmail.inline(fig_name))

  # mail_cont.append("wave-small wave")
  # for code, pivots in tqdm(wave_low_dcit.items()):
  #   data_utils.show_stock_data_eastmoney(code, df_dict[code], save_dir= save_dir, predix="wl_")
  #   fig_name = save_dir +"wl_"+ code+".png"
  #   mail_cont.append(yagmail.inline(fig_name))

  # mail_cont.append("wave-big wave")
  # for code, pivots in tqdm(wave_high_dict.items()):
  #   data_utils.show_stock_data_eastmoney(code, df_dict[code], save_dir= save_dir, predix="wh_")
  #   fig_name = save_dir + "wh_" + code+".png"
  #   mail_cont.append(yagmail.inline(fig_name))

  # mail_cont.append("weekly_wave ")
  # for code in tqdm(weekly_wave_res):
  #   data_utils.show_stock_data_eastmoney(code, weekly_df_dict[code], save_dir= save_dir, predix="weekklywave_", days=100*6)
  #   fig_name = save_dir + "weely_" + code+".png"
  #   mail_cont.append(yagmail.inline(fig_name))
  print(f"****hor common_keys {commonKeys(daily_horizon_dict, weekly_horizon_dict)}")
  print(f"****slope common_keys {commonKeys(daily_slope_dict, weekly_slope_dict)}")
    
  mail_cont.append("daily_horizon ")
  for i, (code, start_date) in tqdm(enumerate(daily_horizon_dict.items())):
    plt.subplot(2,1,1)
    data_utils.show_stock_data_eastmoney(code, df_dict[code], save_dir= save_dir, predix="daily_horizon_", days=150)
    plt.subplot(2,1,2)
    plotHorSupport(code, df_dict[code], support_price_dict, save_dir, "daily", i)
    # plt.show()
    
    fig_name = save_dir + "daily_horizon_" + code+".png"
    mail_cont.append(yagmail.inline(fig_name))  
    
  mail_cont.append("weekly_horizon ")
  for code, start_date in tqdm(weekly_horizon_dict.items()):
    data_utils.show_stock_data_eastmoney(code, weekly_df_dict[code], save_dir= save_dir, predix="weekly_horizon_", days=100*6)
    fig_name = save_dir + "weekly_horizon_" + code+".png"
    mail_cont.append(yagmail.inline(fig_name))
    
  mail_cont.append("daily_slope ")
  for code, start_date in tqdm(daily_slope_dict.items()):
    data_utils.show_stock_data_eastmoney(code, df_dict[code], save_dir= save_dir, predix="daily_slope_", days=150)
    fig_name = save_dir + "daily_slope_" + code+".png"
    mail_cont.append(yagmail.inline(fig_name))    
    
  mail_cont.append("weekly_slope ")
  for code, start_date in tqdm(weekly_slope_dict.items()):
    data_utils.show_stock_data_eastmoney(code, weekly_df_dict[code], save_dir= save_dir, predix="weekly_slope_", days=100*6)
    fig_name = save_dir + "weekly_slope_" + code+".png"
    mail_cont.append(yagmail.inline(fig_name))
  
  # post to mail
  yag.send(mail_send_list, 'subject', mail_cont)

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

