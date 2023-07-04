# 20230704
# python 3.8+

import akshare as ak
import numpy as np

# 下载A股所有数据
# all_df = ak.stock_zh_a_spot_em()
# print(all_df.head())


# 下载个股日k数据图
df_daily = ak.stock_zh_a_hist(symbol="000338", period = "daily", start_date= "20220101", end_date="20230704")
print(df_daily[-7:])


