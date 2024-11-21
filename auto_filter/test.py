import cv2
import numpy as np
import akshare as ak
import pandas as pd

order_num=3
cap_array = np.array([3., 5, 7, 2, 4.7])
sorted_indices = np.argsort(cap_array)[::-1]
aa = cap_array[sorted_indices.astype(int)[:order_num]]

pass
