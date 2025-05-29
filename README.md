
# Google Stock Price Prediction

This project predicts Google stock prices using various machine learning models, including Random Forest, GRU (Gated Recurrent Unit), and Exponential Smoothing. The GRU model outperforms others in terms of accuracy and is used for forecasting future stock prices.

## Features
- Data preprocessing and feature engineering
- Model training and evaluation (Random Forest, GRU, Exponential Smoothing)
- Forecasting future stock prices
- Visualization of actual vs predicted prices
- Streamlit app for interactive predictions

## Installation
1. Clone the repository.
2. Install the required dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage
1. Run the Streamlit app for interactive predictions:

```bash
streamlit run app.py
```

2. Use the GRU model (`gru_model.h5`) for predictions in other scripts.

## Files
- `googl_daily_prices.csv`: Dataset containing historical Google stock prices.
- `gru_model.h5`: Trained GRU model.
- `app.py`: Streamlit app for stock price prediction.
- `requirements.txt`: List of dependencies.

## Results
The GRU model achieved the following performance metrics:
- Mean Absolute Error (MAE): 0.0102
- Mean Squared Error (MSE): 0.0005
- R-squared (R²): 0.9966

## Forecast
The GRU model provides a 30-step forecast for future Google stock prices.

## License
This project is licensed under the MIT License.
