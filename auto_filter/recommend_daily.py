
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
import warnings
warnings.filterwarnings("ignore")

yag = yagmail.SMTP( user="dapaier1115@163.com", password="AXCKERNKNVWSNLVJ", host='smtp.163.com')
mail_send_list = ["liuyao199111@163.com"]


daily_pickle_path = "./sec_data/daily.pickle"
weekly_pickle_path = "./sec_data/weekly.pickle"
monthly_pickle_path = "./sec_data/monthly.pickle"


def dailyTechFilterAndPost():
  df_dict = data_utils.updateToLatestDay(daily_pickle_path, "daily", 1)
  code_dict = raiseLimitTwo(df_dict)
  print(f"raiseLimitTwo size {code_dict.__len__()} {code_dict.keys()}")

  wave_low_dcit, wave_high_dict = waveTechFilter(df_dict)
  print(f"wave_low_dcit size {wave_low_dcit.__len__()} {wave_low_dcit.keys()}")
  print(f"wave_high_dict {wave_high_dict.__len__()} {wave_high_dict.keys()}")

  # save pic
  mail_cont = ["two raise limit in bottom",]
  print(f"save pic")
  save_dir = utils.getProjectPath("auto_filter")+ "/workdata/"
  for file in os.listdir(save_dir):
    if file.endswith('.png'):
      os.remove(save_dir+file)

  for code, data_idx in tqdm(code_dict.items()):
    data_utils.show_stock_data_eastmoney(code, df_dict[code], save_dir= save_dir, predix="tr_")
    fig_name = save_dir+"tr_" + code+".png"
    mail_cont.append(yagmail.inline(fig_name))

  mail_cont.append("wave-small wave")
  for code, pivots in tqdm(wave_low_dcit.items()):
    data_utils.show_stock_data_eastmoney(code, df_dict[code], save_dir= save_dir, predix="wl_")
    fig_name = save_dir +"wl_"+ code+".png"
    mail_cont.append(yagmail.inline(fig_name))

  mail_cont.append("wave-big wave")
  for code, pivots in tqdm(wave_high_dict.items()):
    data_utils.show_stock_data_eastmoney(code, df_dict[code], save_dir= save_dir, predix="wh_")
    fig_name = save_dir + "wh_" + code+".png"
    mail_cont.append(yagmail.inline(fig_name))
  
  # post to mail
  # yag.send(mail_send_list, 'subject', mail_cont)

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

