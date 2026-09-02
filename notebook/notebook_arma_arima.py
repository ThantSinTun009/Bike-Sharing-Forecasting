import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

#%%

# ======================
# Load data
# ======================

df = pd.read_csv("hour.csv")

df['dteday'] = pd.to_datetime(df['dteday'])

df['datetime'] = (
    df['dteday']
    + pd.to_timedelta(df['hr'], unit='h')
)

df = df.set_index('datetime')

ts = df['cnt'].astype(float)

features = df[
    [
        'temp',
        'hum',
        'windspeed',
        'hr',
        'weekday',
        'workingday',
        'holiday',
        'season'
    ]
].astype(float)

#%% 
# ======================
# Train/Test split FIRST
# ======================

train_size = int(len(ts)*0.8)

train = ts.iloc[:train_size]
test = ts.iloc[train_size:]

train_features = features.iloc[:train_size]
test_features = features.iloc[train_size:]

#%%

# ======================
# Stationarity
# ======================

result = adfuller(train)

print("ADF statistic:",result[0])
print("p-value:",result[1])


'''Decision:

p < 0.05 → stationary
p > 0.05 → differencing needed'''


#%%

# ======================
# Differencing visualization
# ======================

train_diff = train.diff().dropna()

plt.figure(figsize=(12,5))
plt.plot(train_diff)
plt.title("Differenced Series")
plt.show()

#%%

# ======================
# ACF/PACF
# ======================

plot_acf(train_diff,lags=40)
plot_pacf(train_diff,lags=40)

plt.show()

#%%

# ======================
# ARMA
# ======================

arma = ARIMA(
    train,
    order=(2,0,2)
)

arma_fit = arma.fit()

#%%

# ======================
# ARIMA
# ======================

arima = ARIMA(
    train,
    order=(2,1,2)
)

arima_fit = arima.fit()

#%%

# ======================
# Forecast
# ======================

forecast = arima_fit.forecast(
    steps=len(test),
    exog=test_features
)

#%%

# ======================
# Plot
# ======================

plt.figure(figsize=(15,6))

plt.plot(test.index,test)
plt.plot(test.index,forecast)

plt.legend(
    ["Actual","Forecast"]
)

plt.title("ARIMA Forecast")
plt.show()

#%%

# ======================
# Metrics
# ======================

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

print("MAE:",mae)
print("RMSE:",rmse)
print("R2 score:", r2)
