
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

import matplotlib.pyplot as plt

import data_utils
from tech.low_raiselimit_two import raiseLimitTwo
from tech.wave_raise import filter_low_wave, filter_high_wave

yag = yagmail.SMTP( user="dapaier1115@163.com", password="AXCKERNKNVWSNLVJ", host='smtp.163.com')
mail_send_list = ["liuyao199111@163.com"]


daily_pickle_path = "./sec_data/daily.pickle"
weekly_pickle_path = "./sec_data/weekly.pickle"
monthly_pickle_path = "./sec_data/monthly.pickle"


def dailyTechFilterAndPost():
  df_dict = data_utils.updateToLatestDay(daily_pickle_path, "daily")
  res_dict = raiseLimitTwo(df_dict)
  print(f"res {res_dict.keys()}")


  # save pic
  print(f"save pic")
  for code, data_idx in tqdm(res_dict.items()):
    data_utils.show_stock_data_eastmoney(code, df_dict[code])

    contents = ['This is the body, and here is just text http://somedomain/image.png',
            'You can find an audio file attached.', '/local/path/song.mp3']

  # post to mail
  yag.send(mail_send_list, 'subject', contents)

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

