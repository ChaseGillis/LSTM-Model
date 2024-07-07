import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf

# Define time_step consistent with training
time_step = 30

# Fetch historical data for input (assuming it's the latest data)
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=730)
ticker_symbol = 'BTC-USD'

# Preprocess data for prediction using the same scaler from training
scaler = MinMaxScaler(feature_range=(0, 1))

# Fetch historical data
btc_data = yf.download(ticker_symbol, start=start_date, end=end_date)
closing_prices = btc_data['Close']

# Preprocess data for prediction using the same scaler from training
closing_prices_scaled = scaler.fit_transform(np.array(closing_prices).reshape(-1, 1))

# Load the saved model
model = load_model('btc_lstm_model.keras')

# Predict ROI for the next 7 days
inputs = closing_prices_scaled[-time_step:]
inputs = inputs.reshape(1, time_step, 1)

predicted_prices = []
for i in range(7):
    prediction = model.predict(inputs).item()
    predicted_prices.append(prediction)
    inputs = np.concatenate((inputs[:, 1:, :], np.array([[prediction]])[:, np.newaxis, :]), axis=1)

# Inverse transform predicted prices to get actual prices
predicted_prices = np.array(predicted_prices).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Calculate predicted ROIs (example calculation, adjust as needed)
predicted_roi = []
for i in range(len(predicted_prices) - 1):
    roi = (((predicted_prices[i + 1] - predicted_prices[i]) / predicted_prices[i]) * 100) * 100
    predicted_roi.append(roi)

# Print predicted ROIs
print("Predicted ROIs for the next 7 days:")
for i, roi in enumerate(predicted_roi, start=1):
    print(f"Day {i}: {roi[0]:.3f}%")

# Determine forecasted state based on the last predicted ROI
forecasted_roi = predicted_roi[-1][0]

if forecasted_roi <= -8:
    forecasted_state = "Strong Bear"
elif -8 < forecasted_roi <= 0:
    forecasted_state = "Weak Bear"
elif 0 < forecasted_roi <= 6:
    forecasted_state = "Choppy"
elif 6 < forecasted_roi <= 16:
    forecasted_state = "Weak Bull"
else:
    forecasted_state = "Strong Bull"

# Print explicitly the forecasted ROI and state
print(f"\nPredicted ROI percent for the next 7 days: {forecasted_roi:.3f}%")
print(f"Forecasted state based on ROI: {forecasted_state}")

# Plot predicted ROI
last_date = btc_data.index[-1]
next_week_dates = pd.date_range(last_date + timedelta(days=1), periods=7)

predicted_roi_extended = np.append(predicted_roi, [None])

plt.figure(figsize=(14, 7))
plt.plot(next_week_dates, predicted_roi_extended, label='Predicted ROI', color='green')
plt.title('Bitcoin Predicted ROI for the Next 7 Days using LSTM')
plt.xlabel('Date')
plt.ylabel('ROI (%)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(next_week_dates, predicted_prices, label='Predicted Prices', color='green')
plt.title('Bitcoin Predicted Prices for the Next 7 Days using LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
