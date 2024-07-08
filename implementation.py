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
inputs = closing_prices_scaled[-time_step:]  # Use the last `time_step` data points
inputs = inputs.reshape(1, time_step, 1)  # Reshape to match model's input shape

predicted_prices = []
current_input = inputs  # Start with the initial input

for _ in range(7):  # Predict for 7 days
    # Predict the next day's price
    next_day_prediction = model.predict(current_input)
    
    # Append the predicted price to the list
    predicted_prices.append(next_day_prediction[0, 0])  # Access the predicted value
    
    # Update current_input to include the new prediction
    current_input = np.roll(current_input, -1, axis=1)  # Shift values to the left
    current_input[0, -1, 0] = next_day_prediction  # Update the last value with prediction

predicted_prices = np.array(predicted_prices).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Calculate predicted ROIs
predicted_roi = []

for i in range(len(predicted_prices) - 1):
    roi = (((predicted_prices[i + 1] - predicted_prices[i]) / predicted_prices[i]) * 100)
    predicted_roi.append(roi)
print(f"Last Price: {closing_prices.iloc[-1]}")
# Calculate ROI for the first day based on observed and first predicted price
initial_roi = (((predicted_prices[0] - closing_prices.iloc[-1]) / closing_prices.iloc[-1]) * 100)
predicted_roi.insert(0, initial_roi)  # Insert initial ROI at the beginning of the list

# Print predicted ROIs and prices
print("Predicted ROIs for the next 7 days:")
for i, roi in enumerate(predicted_roi, start=1):
    print(f"Day {i}: {roi[0]:.3f}%")

print()
print("Predicted Prices for the next 7 days:")
for i, price in enumerate(predicted_prices, start=1):
    print(f"Day {i}: ${price[0]:.2f}")

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

plt.figure(figsize=(14, 7))
plt.plot(next_week_dates, predicted_roi, label='Predicted ROI', color='green')
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

plt.figure(figsize=(14, 7))
plt.plot(closing_prices.index, closing_prices, label='Closing Prices', color='green')
plt.title('Bitcoin Prices Over the Last 2 Years')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()