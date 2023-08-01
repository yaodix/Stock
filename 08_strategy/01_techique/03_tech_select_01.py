import akshare as ak
import numpy as np
import pandas as pd
import datetime as dt
from  tqdm import tqdm

import matplotlib.pyplot as plt


def plot_prices(X, pivots):
    plt.xlim(0, len(X))
    plt.ylim(X.min()*0.99, X.max()*1.01)
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    plt.scatter(pivots.keys(), X[pivots.keys()], color='g')


# 选取股价处于低位的股票
def test(days_after_max_price_thresh = 250, cur_bump_days_thresh = 40):
  # code = "600640"  20191201(最高点)--20210218(最低点)--20211109(启涨点)
  '''
  1. 找出最高价格位置点
  2. 最高价位置点到最近的时间长度， 1年以上？
  3. 拟合价格曲线，判断走势--低位横盘一段时间
    3.1 低位的比例
  4. 出现最近最低价
  
  '''
  code = '600640'
  end_day = '20211109'
  start_day = '20191201'
  
  df_daily = ak.stock_zh_a_hist(symbol=code, period = "daily", start_date=start_day, end_date= end_day, adjust= 'qfq')
  pivots = {}
  # print(f'close {df_daily.head()}')
  close_daily = df_daily.最低
  increase_daily = df_daily.涨跌幅
  low_price_daily = df_daily.收盘
  # 1.
  idx_max = close_daily.idxmax()
  max_close_price = close_daily.max()
  print(f"max_close_price {max_close_price}, day {df_daily.loc[idx_max]['日期']}")
  pivots[idx_max] = max_close_price

  # 2. 
  idx_min = close_daily[idx_max:].idxmin()
  min_close_price = close_daily[idx_max:].min()
  print(f"min_close_price {min_close_price}, day {df_daily.loc[idx_min]['日期']}")
  pivots[idx_min] = min_close_price

  days_after_max_price = idx_min - idx_max
  print(f"days_after_max_price {days_after_max_price}")

  # 跌超过一办， 跌一段时间
  if days_after_max_price < days_after_max_price_thresh and min_close_price / max_close_price > 0.5:
    return

  # 
  if min(idx_min+5, close_daily.size) == close_daily.size:
    return

  idx_min_near = close_daily[min(idx_min+5, close_daily.size):].idxmin()
  min_close_price_near = close_daily[min(idx_min+5, close_daily.size):].min()
  print(f"min_close_price_near {min_close_price_near}, day {df_daily.loc[idx_min_near]['日期']}")
  pivots[idx_min_near] = min_close_price_near

# 条件： 反弹一段时间，最后反弹幅度小
  rat = (min_close_price_near - min_close_price)/ min_close_price
  if idx_min_near- idx_min < cur_bump_days_thresh or \
     rat > 0.08:
    return

  # 反弹区间有涨幅
  idx_max_near = close_daily[idx_min:idx_min_near].idxmax()
  max_close_price_near = close_daily[idx_min:idx_min_near].max()
  print(f"max_close_price_near {max_close_price_near}, day {df_daily.loc[idx_max_near]['日期']}")
  pivots[idx_max_near] = max_close_price_near

  if max_close_price_near / min_close_price < 0.4:
    return
  
  # 最后时间点发生在最近
  if idx_min_near > close_daily.size-15:
    print(f"-------------------------------------")
    print(f"code is {code}")


  # plot_prices(close_daily, pivots)
  # plt.show()


  




if __name__ == "__main__":
  test()