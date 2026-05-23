import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

#%%

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("hour.csv")

# Create datetime column
df['dteday'] = pd.to_datetime(df['dteday'])

df['datetime'] = (
    df['dteday']
    + pd.to_timedelta(df['hr'], unit='h')
)

# Set datetime index
df = df.set_index('datetime')

# Target variable
ts = df['cnt'].astype(float)

# Exogenous variables
features = df[
    ['temp', 'hum', 'windspeed']
].astype(float)

#%%

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

train_size = int(len(ts) * 0.8)

train = ts.iloc[:train_size]
test = ts.iloc[train_size:]

train_features = features.iloc[:train_size]
test_features = features.iloc[train_size:]

# =====================================================
# REDUCE TRAIN SIZE
# (Prevents memory problems)
# =====================================================

train = train.iloc[-5000:]
train_features = train_features.iloc[-5000:]

#%%
# =====================================================
# STATIONARITY TEST
# =====================================================

result = adfuller(train)

print("\n===== ADF TEST =====")
print("ADF Statistic :", result[0])
print("p-value       :", result[1])

if result[1] < 0.05:
    print("Series is stationary")
else:
    print("Series is NOT stationary")

#%%

# =====================================================
# VISUALIZE SERIES
# =====================================================

plt.figure(figsize=(14, 5))

plt.plot(train)

plt.title("Training Series")
plt.xlabel("Time")
plt.ylabel("Bike Count")

plt.show()

#%% 

# =====================================================
# ACF / PACF
# IMPORTANT:
# Since series is stationary,
# use ORIGINAL series (NOT differenced)
# =====================================================

plot_acf(train, lags=40)
plt.show()

plot_pacf(train, lags=40)
plt.show()

#%%

# =====================================================
# SARIMAX MODEL
# =====================================================
#
# Non-seasonal:
# p=1 d=0 q=1
#
# Seasonal:
# P=1 D=0 Q=1 s=24
#
# s=24 because hourly data
# with daily seasonality
#
# =====================================================

model = SARIMAX(
    train,

    exog=train_features,

    order=(1, 0, 1),

    seasonal_order=(1, 0, 1, 24),

    enforce_stationarity=False,
    enforce_invertibility=False
)

#%% 

# =====================================================
# FIT MODEL
# =====================================================

print("\n===== TRAINING MODEL =====")

model_fit = model.fit(
    disp=False
)

print(model_fit.summary())

#%%

# =====================================================
# FORECAST
# =====================================================

forecast = model_fit.forecast(
    steps=len(test),

    exog=test_features
)

#%%

# =====================================================
# PLOT FORECAST
# =====================================================

plt.figure(figsize=(15, 6))

plt.plot(
    test.index,
    test,
    label="Actual"
)

plt.plot(
    test.index,
    forecast,
    label="Forecast"
)

plt.title("SARIMAX Forecast")

plt.xlabel("Time")
plt.ylabel("Bike Count")

plt.legend()

plt.show()


# =====================================================
# EVALUATION
# =====================================================

mae = mean_absolute_error(
    test,
    forecast
)

rmse = np.sqrt(
    mean_squared_error(
        test,
        forecast
    )
)

from sklearn.metrics import r2_score

r2 = r2_score(test, forecast)


print("\n===== METRICS =====")
print("MAE  :", mae)
print("RMSE :", rmse)
print("R2 Score :", r2)


# =====================================================
# RESIDUAL ANALYSIS
# =====================================================

residuals = model_fit.resid

plt.figure(figsize=(14, 5))

plt.plot(residuals)

plt.title("Residuals")

plt.show()

# Residual ACF
plot_acf(
    residuals,
    lags=40
)

plt.show()