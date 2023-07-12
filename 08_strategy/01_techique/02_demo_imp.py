import akshare as ak
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 下载A股所有数据
# all_df = ak.stock_zh_a_spot_em()
# print(all_df.head())

# 20230131-20230531,　形态, 83周期
# target: 600640，20230303 - 20030705

# 下载个股日k数据图
df_daily_all = ak.stock_zh_a_hist(symbol="002747", period = "daily", start_date= "20190101", end_date="20230609")
# df_daily_all = ak.stock_zh_a_hist(symbol="600103", period = "daily", start_date= "20190101", end_date="20230712")
# print(df_daily.info())
print(df_daily_all.info())
# print(df_daily.tail())

price_df = df_daily_all[["收盘"]]
vol_df = df_daily_all["成交量"]


data = np.asarray(price_df)

def plot_pivots(X, pivots):
    plt.xlim(0, len(X))
    plt.ylim(X.min()*0.99, X.max()*1.01)
    plt.plot(np.arange(len(X)), X, 'b:', alpha=0.5)
    # plt.plot(np.arange(len(X)), pivots, 'r-', alpha=0.5)
    # plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')


plot_pivots(data, {})
# plt.show()

training_data_len = int(np.ceil( len(data) * 0.9 ))
scaler = MinMaxScaler(feature_range=(0,1))
# data = data[:, np.newaxis]
scaled_data = scaler.fit_transform(data)

# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape

from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))



# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test


'''simple way but no elegant'''

x_test = scaled_data.copy()
x_test[training_data_len:] = 0

for i in range(training_data_len, len(data) - 60):
    x_test = (x_test[i-60 : i])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    x_test[i, ] = predictions[-1]
    
# Get the root mean squared error (RMSE)
# rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
# print(f"rmse {rmse}")

# Plot the data
train = price_df[:training_data_len]
valid = price_df[training_data_len:]
valid['预测'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

plt.plot(np.array(train['收盘']))
plt.plot(valid[['收盘', '预测']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()