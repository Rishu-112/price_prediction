import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import math
from sklearn.metrics import mean_squared_error

# Custom CSS for UI enhancements
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stTextInput, .stSlider, .stButton {
        margin-bottom: 15px;
    }
    .main {
        padding: 25px;
    }
    .st-selectbox, .st-button, .st-radio, .st-slider, .st-textarea, .st-markdown, .st-table {
        font-family: 'Arial', sans-serif;
    }
    .st-header, .st-subheader {
        color: #4C4F5A;
        font-weight: bold;
        font-size: 24px;
    }
    .st-button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    .st-button:hover {
        background-color: #45a049;
    }
    .st-selectbox, .st-slider, .st-textarea {
        font-size: 16px;
        background-color: #e8f4f8;
        border-radius: 5px;
    }
    .st-markdown {
        color: #4C4F5A;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the LSTM model
model = load_model('Stock Prediction.keras')

# Pre-defined list of popular stock symbols
stock_symbols = ['AAPL', 'GOOG', 'AMZN', 'TSLA', 'MSFT', 'NFLX', 'META', 'NVDA', 'SPY', 'BABA', 'WMT', 'INTC', 'IBM', 'GOOGL', 'BA', 'DIS', 'PYPL', 'AMD']

# Set up Streamlit app title and user input for stock symbol
st.sidebar.header('📊 Daily Stock Price Predictor')

# Dropdown with search hints
stock = st.sidebar.selectbox(
    '🔍 Choose Stock Symbol',
    stock_symbols,
    index=0
)

start = '2012-01-01'
end = dt.now().date()

# Fetch stock data using yfinance
data = yf.download(stock, start, end)
data.reset_index(inplace=True)
data.columns = data.columns.droplevel(1)

# Display stock data
st.subheader(f"📈 Stock Data for {stock}")
st.write(data)

# Calculate moving averages
data['MA50'] = data['Close'].rolling(window=50, min_periods=0).mean()
data['MA200'] = data['Close'].rolling(window=200, min_periods=0).mean()

st.subheader('📊 Original Historical Data')
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=('Price Chart', 'Volume'),
    row_width=[0.2, 0.7]
)

# Add candlestick trace
fig.add_trace(
    go.Candlestick(
        x=data["Date"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
    ),
    row=1,
    col=1
)

# Add moving average traces
fig.add_trace(
    go.Scatter(
        x=data["Date"],
        y=data["MA50"],
        marker_color='grey',
        name="MA50"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=data["Date"],
        y=data["MA200"],
        marker_color='lightgrey',
        name="MA200"
    ),
    row=1,
    col=1
)

# Add volume trace
fig.add_trace(
    go.Bar(
        x=data["Date"],
        y=data['Volume'],
        marker_color='red',
        showlegend=False
    ),
    row=2,
    col=1
)

# Crosshair feature
fig.add_trace(
    go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(color='blue', width=1, dash='dash'),
        showlegend=False,
        hoverinfo='none'
    ),
    row=1, col=1
)

# Update layout settings
fig.update_layout(
    title='📅 Historical Price Chart',
    xaxis_tickfont_size=12,
    yaxis=dict(
        title=dict(text='Price', font=dict(size=14)),
        tickfont=dict(size=12),
    ),
    autosize=False,
    width=800,
    height=600,
    margin=dict(l=50, r=50, b=100, t=100, pad=4),
    paper_bgcolor='white',
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True, key='historical_chart')

# splitting data in train test
data.dropna(inplace=True)
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

x_test = []
y_test = []

for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i-100:i])
    y_test.append(data_test_scale[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_test_predict = model.predict(x_test)
y_test_predict = scaler.inverse_transform(y_test_predict)
y_test = np.array(y_test).reshape(-1, 1)
y_test = scaler.inverse_transform(y_test)

# data creating for second graph
data_train_scale = scaler.fit_transform(data_train)
x_train = []
y_train = []

for i in range(100, data_train_scale.shape[0]):
    x_train.append(data_train_scale[i-100:i])
    y_train.append(data_train_scale[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

y_train_predict = model.predict(x_train)
y_train_predict = scaler.inverse_transform(y_train_predict)
y_train = np.array(y_train).reshape(-1, 1)
y_train = scaler.inverse_transform(y_train)

df1 = pd.DataFrame(y_train_predict, columns=["Prediction"])
df2 = pd.DataFrame(y_test_predict, columns=["Prediction"])
df = pd.DataFrame({'Prediction': [np.nan] * 100})
df12 = pd.concat([df1, df2], ignore_index=True)
df_final = pd.concat([df, df12], ignore_index=True)

data = data.reset_index(drop=True)
df_final = df_final.reset_index(drop=True)
combined_df = pd.concat([data, df_final], axis=1)

# accuracy and error
mape = mean_absolute_percentage_error(combined_df.Close[101:], combined_df.Prediction[101:])
accuracy = 100 - mape

mse = math.sqrt(mean_squared_error(combined_df.Close[101:], combined_df.Prediction[101:]))

# Second graph
st.subheader('📉 Original Price vs Predicted Price')
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=combined_df.index,
    y=combined_df['Close'],
    mode='lines',
    name='Actual Close Price',
    line=dict(color='red'),
))
fig.add_trace(go.Scatter(
    x=combined_df.index,
    y=combined_df['Prediction'],
    mode='lines',
    name='Predicted Price',
    line=dict(color='green'),
))

fig.update_layout(
    title='📉 Stock Price Prediction',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=True,
    template='plotly_white'
)
st.plotly_chart(fig, use_container_width=True, key='chart2')

st.subheader('🔮 Future Stock Price Predictions')
st.write("Here are the predictions for the next 100 business days:")

# Future prediction
last_100_days = data[-100:]
last_100_days_scaled = scaler.transform(last_100_days[['Close']])

predictions = []
current_input = last_100_days_scaled.reshape((1, 100, 1))

for _ in range(100):
    next_prediction = model.predict(current_input)
    predictions.append(next_prediction[0, 0])
    current_input = np.append(current_input[:, 1:, :], next_prediction.reshape(1, 1, 1), axis=1)

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Bridge the Gap (Connect Green & Red Lines)
historical_prices = data[['Date', 'Close']].copy()
historical_prices.set_index('Date', inplace=True)
last_date = historical_prices.index[-1]
last_close = historical_prices['Close'].iloc[-1]

# Prepend the last historical close price to the predictions array
predictions_with_start = np.insert(predictions, 0, last_close)

# Generate business dates starting from the exact last historical date
predicted_dates_with_start = pd.date_range(start=last_date, periods=len(predictions_with_start), freq='B')

predicted_prices_df = pd.DataFrame(
    data=predictions_with_start,
    index=predicted_dates_with_start,
    columns=['Predicted Price']
)

# Third graph
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=historical_prices.index,
    y=historical_prices['Close'],
    mode='lines',
    name='Historical Close Price',
    line=dict(color='green'),
))
fig.add_trace(go.Scatter(
    x=predicted_prices_df.index,
    y=predicted_prices_df['Predicted Price'],
    mode='lines',
    name='Predicted Price',
    line=dict(color='red'),
))

fig.update_layout(
    title='📅 Future Stock Price Predictions',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=True,
)
st.plotly_chart(fig, use_container_width=True, key='chart3')

st.markdown(f"<div class='st-accuracy'><strong>Model Accuracy: </strong>{accuracy:.2f}%</div>", unsafe_allow_html=True)
st.markdown(f"<div class='st-accuracy'><strong>Mean Squared Error (RMSE): </strong>{mse:.2f}</div>", unsafe_allow_html=True)
