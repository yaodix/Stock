
# zigzag示例

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zigzag import *
from pandas_datareader import get_data_yahoo

# This is not nessessary to use zigzag. It's only here so that
# this example is reproducible.

def test_raw_pvp():
  np.random.seed(1997)

  X = np.cumprod(1 + np.random.randn(100) * 0.01)
  pivots = peak_valley_pivots(X, 0.03, -0.03)

  def plot_pivots(X, pivots):
      plt.xlim(0, len(X))
      plt.ylim(X.min()*0.99, X.max()*1.01)
      plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
      plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
      plt.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='g')
      plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')
      

  plot_pivots(X, pivots)
  plt.show()
  
def test_pandas_pvp():
  X = get_data_yahoo('GOOG')['Adj Close']
  pivots = peak_valley_pivots(X.values, 0.2, -0.2)
  ts_pivots = pd.Series(X, index=X.index)
  ts_pivots = ts_pivots[pivots != 0]
  X.plot()
  ts_pivots.plot(style='g-o')
  plt.show()
  
  
  
if __name__ == "__main__":
  test_raw_pvp()
  # test_pandas_pvp()