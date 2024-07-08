import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# Fetch historical data
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=730)
ticker_symbol = 'BTC-USD'
btc_data = yf.download(ticker_symbol, start=start_date, end=end_date)
closing_prices = btc_data['Close']

# Preprocess data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
closing_prices_scaled = scaler.fit_transform(np.array(closing_prices).reshape(-1, 1))

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 30  # Number of time steps to look back
X, y = create_dataset(closing_prices_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape input to be [samples, time steps, features]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Assuming X_train.shape[1] is the number of time steps and 1 is the number of features
input_layer = Input(shape=(X_train.shape[1], 1))
# LSTM layer 1
lstm1 = LSTM(units=100, return_sequences=True)(input_layer)
dropout1 = Dropout(0.2)(lstm1)

# LSTM layer 2
lstm2 = LSTM(units=100, return_sequences=False)(dropout1)
dropout2 = Dropout(0.2)(lstm2)

# Output layer
output_layer = Dense(units=1)(dropout2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM model
model.fit(X_train, y_train, epochs=100, batch_size=32)


# Save the trained model
model.save('btc_lstm_model.keras')
