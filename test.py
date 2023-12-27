

import sys
import akshare as ak
import numpy as np
import pandas as pd
import datetime as dt
from  tqdm import tqdm

import matplotlib.pyplot as plt


stock_individual_info_em_df = ak.stock_individual_info_em(symbol="000001")

print(stock_individual_info_em_df["value"][0])
