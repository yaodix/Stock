import pandas as pd
import numpy as np

index = pd.date_range("1/1/2000", periods=8)
s = pd.Series(np.random.randn(5))


# Series 构建DataFrame
df = pd.DataFrame({'one': pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
                   'two': pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),
                   'three': pd.Series(np.random.randn(3), index=['b', 'c', 'd'])})

# print(df)

# print(df.apply(lambda x: x.max() - x.min()))

# print(df.apply(np.exp))
tsdf = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'],
                 index=pd.date_range('1/1/2023', periods=10))
print(tsdf)

row1 = tsdf.loc['1/1/2023']  # 返回Series类型
row2 = tsdf.loc[['1/1/2023']]  # 返回DataFrame类型

d3 = tsdf.loc[['1/1/2023','1/2/2023']]  # 返回DataFrame类型
d4 = tsdf.loc['1/1/2023':'1/2/2023']  # 返回DataFrame类型

rk = row1.keys()
rv = row1.values

# dataframe 索引
tsdf.iloc[3:7] = np.nan

# print(tsdf["A"].agg(np.sum))


# 构造dataframe
def test1():
  dates = pd.date_range('20230101', periods=6)
  df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
  print(f'df is\n{df}')
  print(f'df describe is\n{df.describe()}')  # 数据的快速统计汇总
  print(f"df index\n{df.index}")
  print(f"df colunmns\n{df.columns}")
  print(f"df values\n{df.values}")
  
  
if __name__ == "__main__":
  # test1()
  print('1')
  