# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#%%

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

# Choose dataset

file_path_2 = "day.csv"
file_path = "hour.csv"

df = pd.read_csv(file_path)

df_2 = pd.read_csv(file_path_2)

# Convert date column
df["dteday"] = pd.to_datetime(df["dteday"])

# Sort by date
df = df.sort_values("dteday").reset_index(drop=True)

# ==========================================================
# 2. FEATURE SELECTION
# ==========================================================

feature_cols = [
    'season',
    'yr',
    'mnth',
    'hr' if 'hr' in df.columns else None,
    'holiday',
    'weekday',
    'workingday',
    'weathersit',
    'temp',
    'atemp',
    'hum',
    'windspeed'
]

# Remove None if using day.csv
feature_cols = [col for col in feature_cols if col is not None]

target_col = "cnt"

#%%

# ==========================================================
# 3. FEATURE ENGINEERING
# ==========================================================

window_size = 7 if "day.csv" in file_path else 24

def mad(x):
    return np.mean(np.abs(x - np.mean(x)))

# Rolling statistics for target variable
df["cnt_roll_mean"] = df[target_col].rolling(window_size).mean()
df["cnt_roll_std"] = df[target_col].rolling(window_size).std()
df["cnt_roll_median"] = df[target_col].rolling(window_size).median()
df["cnt_roll_mad"] = df[target_col].rolling(window_size).apply(mad, raw=True)

# Drop NA values
df = df.dropna().reset_index(drop=True)

# Final feature columns
feature_cols += [
    "cnt_roll_mean",
    "cnt_roll_std",
    "cnt_roll_median",
    "cnt_roll_mad"
]

# ==========================================================
# 4. PREPARE X AND y
# ==========================================================

X_raw = df[feature_cols]

# Predict next timestep
y_raw = df[target_col].shift(-1)

# Remove last row
X_raw = X_raw.iloc[:-1]
y_raw = y_raw.iloc[:-1]

# ==========================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

n = len(X_raw)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train_raw = X_raw.iloc[:train_end]
X_val_raw = X_raw.iloc[train_end:val_end]
X_test_raw = X_raw.iloc[val_end:]

y_train_raw = y_raw.iloc[:train_end]
y_val_raw = y_raw.iloc[train_end:val_end]
y_test_raw = y_raw.iloc[val_end:]

#%%
# ==========================================================
# 6. SCALE DATA
# ==========================================================

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train_scaled = x_scaler.fit_transform(X_train_raw)
X_val_scaled = x_scaler.transform(X_val_raw)
X_test_scaled = x_scaler.transform(X_test_raw)

y_train_scaled = y_scaler.fit_transform(
    y_train_raw.values.reshape(-1,1))

y_val_scaled = y_scaler.transform(
    y_val_raw.values.reshape(-1,1))

y_test_scaled = y_scaler.transform(
    y_test_raw.values.reshape(-1,1))

#%%
# ==========================================================
# 7. CREATE SEQUENCES
# ==========================================================

seq_length = 7 if "day.csv" in file_path else 24

def create_sequences(X, y, seq_length):

    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_length):

        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])

    return np.array(X_seq), np.array(y_seq)

X_train, y_train = create_sequences(
    X_train_scaled,
    y_train_scaled,
    seq_length
)

X_val, y_val = create_sequences(
    X_val_scaled,
    y_val_scaled,
    seq_length
)

X_test, y_test = create_sequences(
    X_test_scaled,
    y_test_scaled,
    seq_length
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

#%%
# ==========================================================
# 8. BUILD MODELS
# ==========================================================

def build_model(model_type, input_shape):

    model = Sequential()

    if model_type == "SimpleRNN":
        model.add(SimpleRNN(
            64,
            activation='tanh',
            input_shape=input_shape
        ))

    elif model_type == "LSTM":
        model.add(LSTM(
            64,
            activation='tanh',
            input_shape=input_shape
        ))

    elif model_type == "GRU":
        model.add(GRU(
            64,
            activation='tanh',
            input_shape=input_shape
        ))

    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model

#%%
# ==========================================================
# 9. TRAIN AND EVALUATE
# ==========================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    y_scaler
):

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_scaled = model.predict(X_test)

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test)

    result = {
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

    return history, y_true, y_pred, result

#%%
# ==========================================================
# 10. TRAIN ALL MODELS
# ==========================================================

input_shape = (X_train.shape[1], X_train.shape[2])

models = {
    "SimpleRNN": build_model("SimpleRNN", input_shape),
    "LSTM": build_model("LSTM", input_shape),
    "GRU": build_model("GRU", input_shape)
}

all_results = []
histories = {}
predictions = {}

for name, model in models.items():

    history, y_true, y_pred, result = evaluate_model(
        name,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        y_scaler
    )

    all_results.append(result)
    histories[name] = history
    predictions[name] = (y_true, y_pred)

#%%
# ==========================================================
# 11. RESULTS COMPARISON
# ==========================================================

result_df = pd.DataFrame(all_results)

print(result_df)

# ==========================================================
# 12. PLOT RESULTS
# ==========================================================

plt.figure(figsize=(10,6))

for name in predictions:

    y_true, y_pred = predictions[name]

    plt.plot(
        y_true[:100],
        label="True"
    )

    plt.plot(
        y_pred[:100],
        label=f"{name} Prediction"
    )

    plt.title(f"{name} Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Bike Rental Count")

    plt.legend()
    plt.show()

# ==========================================================
# 13. VALIDATION LOSS
# ==========================================================

plt.figure(figsize=(10,6))

for name, history in histories.items():

    plt.plot(
        history.history["val_loss"],
        label=name
    )

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt

# Models
models = ["SimpleRNN", "LSTM", "GRU"]

# Metrics (your original values)
mae = [86.79, 94.7, 86.12]
rmse = [120.93, 132.36, 122.59]
r2 = [0.661, 0.616, 0.685]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- MAE ---
axes[0].bar(models, mae)
axes[0].set_title("MAE Comparison")
axes[0].set_xlabel("Model")
axes[0].set_ylabel("MAE")

# --- RMSE ---
axes[1].bar(models, rmse)
axes[1].set_title("RMSE Comparison")
axes[1].set_xlabel("Model")
axes[1].set_ylabel("RMSE")

# --- R² ---
axes[2].bar(models, r2)
axes[2].set_title("R² Comparison")
axes[2].set_xlabel("Model")
axes[2].set_ylabel("R²")

plt.tight_layout()
plt.show()