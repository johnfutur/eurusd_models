import joblib
import numpy as np
import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved model and feature names
model = joblib.load("eurusd_xgb_model.pkl")
feature_names = joblib.load("model_features.pkl")

# Connect to MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    mt5.shutdown()

# Retrieve latest EUR/USD data (5-min timeframe, last 200 candles)
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M5
n_candles = 200

rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
if rates is None:
    print("Failed to retrieve data")
    mt5.shutdown()

# Disconnect from MT5
mt5.shutdown()

# Convert data to DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

# Calculate EMA 200
df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

# Calculate RSI (14)
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['rsi_14'] = 100 - (100 / (1 + rs))

# Calculate MACD (12,26,9)
short_ema = df['close'].ewm(span=12, adjust=False).mean()
long_ema = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = short_ema - long_ema
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

# Calculate Bollinger Bands
df['bollinger_mid'] = df['close'].rolling(window=20).mean()
df['bollinger_std'] = df['close'].rolling(window=20).std()
df['bollinger_upper'] = df['bollinger_mid'] + (df['bollinger_std'] * 2)
df['bollinger_lower'] = df['bollinger_mid'] - (df['bollinger_std'] * 2)

# Drop NaN values
df.dropna(inplace=True)

# Prepare the latest data for prediction
latest_data = df[feature_names].iloc[-1:].values

# Predict the next 10 close prices
future_prices = model.predict(latest_data)[0]  # Extract first row (1D array)

# Get current close price
current_close = df.iloc[-1]['close']

# Determine trade signal based on final predicted price
predicted_final_price = future_prices[-1]

if predicted_final_price > current_close:
    signal = "BUY"
    potential_profit = predicted_final_price - current_close
elif predicted_final_price < current_close:
    signal = "SELL"
    potential_profit = current_close - predicted_final_price
else:
    signal = "HOLD"
    potential_profit = 0

# Print results
print(f"Predicted next 10 EUR/USD prices: {future_prices.tolist()}")
print(f"Trade Signal: {signal}")
print(f"Potential Profit per unit (approx.): {potential_profit:.5f}")

from tradingview_ta import TA_Handler, Interval, Exchange

# Define the asset to analyze
analysis = TA_Handler(
    symbol="EURUSD",
    screener="forex",
    exchange="FX_IDC",
    interval=Interval.INTERVAL_5_MINUTES
)

# Get TradingView analysis
result = analysis.get_analysis()

# Extract buy/sell/neutral counts
buy_count = result.summary['BUY']
sell_count = result.summary['SELL']
neutral_count = result.summary['NEUTRAL']
overall_signal = result.summary['RECOMMENDATION']  # Final trading recommendation

# Print the results
print(f"🔹 BUY Indicators: {buy_count}")
print(f"🔹 SELL Indicators: {sell_count}")
print(f"🔹 NEUTRAL Indicators: {neutral_count}")
print(f"\n🔥 Overall TradingView Signal: {overall_signal}")



# Plot the predicted prices
time_steps = np.arange(1, 11)  # Future time steps
plt.figure(figsize=(10, 5))
plt.plot(time_steps, future_prices, marker='o', linestyle='-', color='b', label='Predicted Prices')
plt.axhline(y=current_close, color='r', linestyle='--', label='Current Close Price')
plt.xlabel('Future Time Steps (5-min intervals)')
plt.ylabel('Price')
plt.title('Predicted EUR/USD Prices for Next 10 Intervals')
plt.legend()
plt.grid()
plt.show()
