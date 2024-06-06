#  斜率K + 单调性 + 支点数量 + 长度?
# find and sort

# filter by slope of pivot

import sys
sys.path.append(r"/home/yao/myproject/Stock/01_basic")
sys.path.append(r"/home/yao/myproject/Stock/00_data")

import akshare as ak
import numpy as np
import matplotlib.pyplot as plt
from  tqdm import tqdm

from my_zigzag import get_pivots, plot_pivots, plot_pivot_line
from stock_data_utils import get_sh_sz_A_name


def filter_pivot_line(src_data, pivots, k_thresh, verbose = False):
  
  low_pivots_index = [k for k, v in pivots.items() if v == -1]
  high_pivots_index = [k for k, v in pivots.items() if v == 1]
  if (len(low_pivots_index) < 3):
    return
  if (high_pivots_index[-1] > low_pivots_index[-1]):
    del high_pivots_index[-1]
    
  k_sup_last1 = (src_data[low_pivots_index[-1]] - src_data[low_pivots_index[-2]]) / (low_pivots_index[-1] - low_pivots_index[-2])
  k_sup_last2 = (src_data[low_pivots_index[-2]] - src_data[low_pivots_index[-3]]) / (low_pivots_index[-2] - low_pivots_index[-3])
  
  k_res_last1 = (src_data[high_pivots_index[-1]] - src_data[high_pivots_index[-2]]) / (high_pivots_index[-1] - high_pivots_index[-2])
  if verbose:
    print(f"k_sup_last1 {k_sup_last1}, k_sup_last2 {k_sup_last2}, k_res_last1 {k_res_last1}")

  if k_thresh < k_sup_last1  and  k_thresh < k_sup_last2  and  \
     k_thresh < k_res_last1 and k_res_last1 < max(k_sup_last1, k_sup_last2)+0.1 and \
     1 < abs(low_pivots_index[-1] - high_pivots_index[-1]) and  abs(low_pivots_index[-1] - high_pivots_index[-1]) < 15 and \
     1 < abs(low_pivots_index[-2] - high_pivots_index[-1]) and  abs(low_pivots_index[-2] - high_pivots_index[-1]) < 15 and \
     1 < abs(low_pivots_index[-2] - high_pivots_index[-2]) and  abs(low_pivots_index[-2] - high_pivots_index[-2]) < 15 and \
     1 < abs(low_pivots_index[-3] - high_pivots_index[-2]) and  abs(low_pivots_index[-3] - high_pivots_index[-2]) < 15 and \
     True:
    return True
  

if __name__ == "__main__":
  stocks = get_sh_sz_A_name()
  for code in tqdm(stocks.code.tolist()):
    # df_daily = ak.stock_zh_a_hist(symbol="002952", period = "daily", start_date= "20230101", end_date="20240531")
    df_daily = ak.stock_zh_a_hist(symbol=code, period = "daily", start_date= "20230102", end_date="20241215")
    
    X = df_daily["收盘"]
    data = np.asarray(X)    
    pivots = get_pivots(data, 0.06, 0.06)
    
    sel =  filter_pivot_line(data, pivots, 0.05)
    # sel = True
    if sel:
      print(code)
      plt.clf()
      plot_pivots(data, pivots)
      plot_pivot_line(data, pivots)
      plt.savefig('./workdata/'+code + '.jpg')
      # break
      # plt.show()
