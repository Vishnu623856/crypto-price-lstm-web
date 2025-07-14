import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

st.title("📦 Crypto Price Prediction with LSTM")

coin = st.sidebar.selectbox("Choose Cryptocurrency", ["BTC-USD", "ETH-USD"])
start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

@st.cache_data
def load_data(symb, start, end):
    return yf.download(symb, start=start, end=end)[['Close']].dropna()

data = load_data(coin, start, end)
st.subheader(f"{coin} Price Data")
st.line_chart(data['Close'])

# Prepare data
scaler = joblib.load("scaler.save")
seq_len = 60
scaled = scaler.transform(data[['Close']].values)
X = []
for i in range(seq_len, len(scaled)):
    X.append(scaled[i-seq_len:i, 0])
X = np.array(X).reshape(-1, seq_len, 1)

model = load_model("lstm_model.h5")
pred = model.predict(X)
predicted_prices = scaler.inverse_transform(pred)
actual_prices = data['Close'].values[seq_len:]

# Plot
st.subheader("Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual_prices, label="Actual")
ax.plot(predicted_prices.flatten(), label="Predicted")
ax.legend()
st.pyplot(fig)
