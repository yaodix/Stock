

def test_one():
  df_daily = ak.stock_zh_a_hist(symbol="000338", period = "daily", start_date= "20230131", end_date="20230531")
  t_df_daily = ak.stock_zh_a_hist(symbol="600640", period = "daily", start_date= "20230302", end_date="20230704")
  print(df_daily.tail())

  X = df_daily["收盘"]
  print(X.tail())


def test_merge():
  df1 = pd.DataFrame(
  {
      "A": ["A0", "A1", "A2", "A3"],
      "B": ["B0", "B1", "B2", "B3"],
      "C": ["C0", "C1", "C2", "C3"],
      "D": ["D0", "D1", "D2", "D3"],
  },
  index=[0, 1, 2, 3],)


df2 = pd.DataFrame(
    {
        "A": ["A4", "A5", "A6", "A7"],
        "B": ["B4", "B5", "B6", "B7"],
        "C": ["C4", "C5", "C6", "C7"],
        "D": ["D4", "D5", "D6", "D7"],
    },
  index=[0, 1, 2, 3],)

frames = [df1, df2]

result = pd.concat(frames, ignore_index=True)
  print(f'merge {result}')
  