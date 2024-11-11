
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
  code_dict = raiseLimitTwo(df_dict)
  print(f"res {code_dict.keys()}")


  # save pic
  raise_fail_cont = ["底部涨停回调",]
  print(f"save pic")
  for code, data_idx in tqdm(code_dict.items()):
    data_utils.show_stock_data_eastmoney(code, df_dict[code])
    fig_name = "/home/yao/workspace/Stock/auto_filter/workdata/" + code+".png"

    # raise_fail_cont.append(df_dict[""])
    raise_fail_cont.append(yagmail.inline(fig_name))

  # post to mail
  yag.send(mail_send_list, 'subject', raise_fail_cont)

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

