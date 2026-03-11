import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

model = load_model('C:/Users/AJAY BHOLA/OneDrive/Desktop/mid project/Model.keras')
start = '2012-01-01'
end = '2022-12-21'

st.title('Stock Trend prediction')

user_input=st.text_input('Enter Stock ticker','TSLA')

df = yf.download('TSLA', start=start, end=end)

# describing data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

#visialization
st.subheader('Closing price vs time chart')
fig =plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs time chart 100MA')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs time chart 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


data_train = pd.DataFrame(df.Close[0: int(len(df)*0.80)])
data_test = pd.DataFrame(df.Close[int(len(df)*0.80): len(df)])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)

# testing part

pas_100_days = data_train.tail(100)
final_df = pd.concat([pas_100_days, data_test], ignore_index=True)
input_data  =  scaler.fit_transform(final_df)

x = []
y = []

for i in range(100, input_data.shape[0]):
    x.append(input_data[i-100:i])
    y.append(input_data[i,0])
x, y = np.array(x), np.array(y)

y_predict = model.predict(x)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predict=y_predict*scale_factor
y=y*scale_factor

#Final graph
st.subheader('Predicton vs Original')
fig2=plt.figure(figsize=(10,8))
plt.plot(y_predict, 'r', label = 'Predicted Price')
plt.plot(y, 'g', label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)
