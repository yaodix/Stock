import pandas as pd
import numpy as np
index = pd.date_range("1/1/2000", periods=8)
s = pd.Series(np.random.randn(5))

df = pd.DataFrame({'one': pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
                   'two': pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),
                   'three': pd.Series(np.random.randn(3), index=['b', 'c', 'd'])})

print(df)

print(df['two'])