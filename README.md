# Bike Sharing Demand Forecasting

This project focuses on predicting hourly bike rental demand using time series forecasting techniques. It compares traditional statistical models and deep learning architectures to evaluate predictive performance.

📊 Problem Statement

Bike-sharing systems require accurate demand forecasting to optimize bike distribution and availability. However, demand is highly dynamic and depends on weather, time, and seasonal factors, making prediction challenging.

📦 Dataset

The dataset used is the Bike Sharing Dataset (hour.csv) which includes:

Hourly bike rental counts (cnt)
Weather conditions
Temperature, humidity, wind speed
Time-based features (hour, day, season)
⚙️ Methodology
Classical Models:
ARMA
ARIMA
SARIMAX
Deep Learning Models:
SimpleRNN
LSTM
GRU
🧠 Feature Engineering
24-hour sliding window
Rolling mean, std, median, MAD
MinMax normalization
One-step ahead forecasting (shift = -1)
🏗️ Model Architecture

All deep learning models use:

64-unit recurrent layer (tanh)
Dropout (0.2)
Dense layer (32 ReLU)
Output layer (linear)
Optimizer: Adam
Loss: MSE

| Model     | MAE   | RMSE   | R² Score |
| --------- | ----- | ------ | -------- |
| SimpleRNN | 86.29 | 122.56 | 0.6716   |
| LSTM      | 93.67 | 130.12 | 0.6298   |
| GRU       | 84.13 | 117.96 | 0.6957   |
