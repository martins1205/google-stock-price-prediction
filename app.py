import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the GRU model
gru_model = load_model('./gru_model.h5')

# Load the dataset
googl_data = pd.read_csv('googl_daily_prices.csv')
googl_data['date'] = pd.to_datetime(googl_data['date'])
googl_data = googl_data.sort_values(by='date').reset_index(drop=True)

# Preprocess the data
features = ['1. open', '2. high', '3. low', '5. volume', 'Previous Day Close', '7-Day Moving Average', '15-Day Moving Average']

def preprocess_data(data):
    data['Previous Day Close'] = data['4. close'].shift(1)
    data['7-Day Moving Average'] = data['4. close'].rolling(window=7).mean()
    data['15-Day Moving Average'] = data['4. close'].rolling(window=15).mean()
    data = data.dropna()
    return data

googl_data = preprocess_data(googl_data)

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_to_normalize = ['1. open', '2. high', '3. low', '4. close', '5. volume', 'Previous Day Close', '7-Day Moving Average', '15-Day Moving Average']
googl_data[features_to_normalize] = scaler.fit_transform(googl_data[features_to_normalize])

# Split the data
train_size = int(len(googl_data) * 0.8)
train_data = googl_data.iloc[:train_size]
test_data = googl_data.iloc[train_size:]

X_test = test_data[features]
X_test_gru = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# Streamlit app
st.title('Google Stock Price Prediction')

# Display the dataset
if st.checkbox('Show Dataset'):
    st.write(googl_data)

# Predict using GRU model
y_pred_gru = gru_model.predict(X_test_gru)

# Visualize predictions
st.subheader('Actual vs Predicted Prices')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_data['date'], test_data['4. close'], label='Actual Prices', color='blue')
ax.plot(test_data['date'], y_pred_gru, label='Predicted Prices (GRU)', color='orange')
ax.set_title('GRU Model Performance: Actual vs Predicted Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Closing Prices')
ax.legend()
st.pyplot(fig)

# Forecast future prices
st.subheader('Forecast Future Prices')
future_steps = st.slider('Select number of future steps to forecast', 1, 60, 30)
forecasted_prices = []
last_known_data = X_test_gru[-1]

for _ in range(future_steps):
    next_price = gru_model.predict(last_known_data.reshape(1, -1, 1), verbose=0)[0, 0]
    forecasted_prices.append(next_price)
    last_known_data = np.roll(last_known_data, -1)
    last_known_data[-1] = next_price

st.write('Forecasted Prices:', forecasted_prices)

# Visualize forecasted prices
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(range(1, future_steps + 1), forecasted_prices, label='Forecasted Prices (GRU)', color='purple')
ax2.set_title('Forecasted Future Google Stock Prices (GRU Model)')
ax2.set_xlabel('Future Time Steps')
ax2.set_ylabel('Normalized Closing Prices')
ax2.legend()
st.pyplot(fig2)