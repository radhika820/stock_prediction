import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from ta.momentum import RSIIndicator

ticker = 'GOOGL'
data = yf.download(ticker, start='2018-01-01', end='2023-12-31')

df = pd.DataFrame()
df['Close'] = data['Close']
df['SMA'] = df['Close'].rolling(window=20).mean()

close_series = df['Close']
if isinstance(close_series, pd.DataFrame):
    close_series = close_series.squeeze()

df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
df.dropna(inplace=True)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['Close', 'SMA', 'RSI']])

X = scaled_features[:-1]
y = scaled_features[1:, 0]
X = X[:len(y)]

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"📊 MSE: {mse:.6f}")
print(f"📊 RMSE: {rmse:.6f}")
print(f"📊 R² Score: {r2:.6f}")

plt.figure(figsize=(14, 6))
plt.plot(range(len(y_test)), y_test, label='Actual', color='blue')
plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='red')
plt.title(f'Stock Price Prediction for {ticker}')
plt.xlabel('Time Index')
plt.ylabel('Normalized Close Price')
plt.legend()
plt.grid(True)
plt.show()

last_input = scaled_features[-1].reshape(1, -1)
predicted_scaled_price = model.predict(last_input)[0]
predicted_price = scaler.inverse_transform([[predicted_scaled_price, 0, 0]])[0][0]

print(f"\n📌 Predicted Next Day's Close Price for {ticker}: ${predicted_price:.2f}")
