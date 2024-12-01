import cv2
import numpy as np
import akshare as ak
import pandas as pd
import matplotlib
import mplfinance as mpf


fig = mpf.figure(figsize=(12,9))
ax1 = fig.add_subplot(2,2,1,style='blueskies')
ax2 = fig.add_subplot(2,2,2,style='yahoo')
 
s = mpf.make_mpf_style(base_mpl_style='fast',base_mpf_style='nightclouds')
ax3 = fig.add_subplot(2,2,3,style=s)
ax4 = fig.add_subplot(2,2,4,style='starsandstripes')
 
mpf.plot(df,ax=ax1,axtitle='blueskies',xrotation=15)
mpf.plot(df,type='candle',ax=ax2,axtitle='yahoo',xrotation=15)
mpf.plot(df,ax=ax3,type='candle',axtitle='nightclouds')
mpf.plot(df,type='candle',ax=ax4,axtitle='starsandstripes')