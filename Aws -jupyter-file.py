!pip install yfinance
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json
import math
import boto3
import os
import shutil
import subprocess


# Sklearn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tensorflow / Keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -------------------------------
# 2) Add Technical Indicators
# -------------------------------
def add_indicators(df):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["EMA7"] = df["Close"].ewm(span=7, adjust=False).mean()
    df["EMA14"] = df["Close"].ewm(span=14, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["BB_MID"] = df["Close"].rolling(20).mean()
    close_std = df["Close"].rolling(20).std().squeeze()
    df["BB_UPPER"] = df["BB_MID"] + 2 * close_std
    df["BB_LOWER"] = df["BB_MID"] - 2 * close_std
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df
# Example run
df = add_indicators(df)
print(df.head((25)))



# -------------------------------
# 1) Fetch Stock Data
# -------------------------------
def get_stock_data(symbol, start_date, interval="1d"):
    original_symbol = symbol
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    end_date = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval, auto_adjust=False)
    if df.empty:
        print(f"⚠ No data found for {symbol}")
        return None, original_symbol
    df = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
    df.rename(columns={"Date": "DATE"}, inplace=True)
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    return df, original_symbol
# Example run
df, stock_name = get_stock_data("INFY", "2021-01-01")
print(df.head((25)))



# -------------------------------
# 4) Prepare Sequences (Multi-output: Open + Close)
# -------------------------------
def prepare_sequences(df, lookback=30, feature_cols=None, target_cols=None):
    if feature_cols is None:
        feature_cols = ["Open", "High", "Low", "Volume", "Close",
                        "RSI", "MACD", "MACD_SIGNAL",
                        "EMA7", "EMA14", "EMA21", "EMA50", "EMA100",
                        "ATR", "BB_MID", "BB_UPPER", "BB_LOWER"]
    if target_cols is None:
        target_cols = ["Open", "Close"]
    # Scale features
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(df[feature_cols])
    # Scale targets (Open + Close)
    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(df[target_cols])
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(scaled_features[i - lookback:i, :])
        y.append(scaled_target[i, :])   # Multi-output
    return np.array(X), np.array(y), feature_scaler, target_scaler, feature_cols, target_cols, df


# -------------------------------
# 5) Build LSTM Model (Multi-output)
# -------------------------------
def build_lstm(input_timesteps, n_features, n_outputs):
    model = Sequential([
        Input(shape=(input_timesteps, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(n_outputs)   # Multi-output: Open + Close
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mean_squared_error")
    return model
# -------------------------------
# 6) Train, Evaluate, Predict (Multi-output)
# -------------------------------
def train_eval_predict_lstm(df, lookback=60, test_size=0.2, future_days=3):

    feature_cols = ["Open", "High", "Low", "Volume", "Close",
                    "RSI", "MACD", "MACD_SIGNAL",
                    "EMA7", "EMA14", "EMA21", "EMA50", "EMA100",
                    "ATR", "BB_MID", "BB_UPPER", "BB_LOWER"]

    target_cols = ["Open", "Close"]

    X, y, feature_scaler, target_scaler, features, targets, df = prepare_sequences(
        df, lookback, feature_cols, target_cols
    )

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm(lookback, X.shape[2], len(targets))

    early_stopping = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    if len(X_test) > 0:
        y_pred = model.predict(X_test, verbose=0)
        y_test_inv = target_scaler.inverse_transform(y_test)
        y_pred_inv = target_scaler.inverse_transform(y_pred)

        print("📏 MAE:", mean_absolute_error(y_test_inv, y_pred_inv))
        print("📉 RMSE:", math.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
        print("📊 R²:", r2_score(y_test_inv, y_pred_inv))

    # Retrain on full data
    model.fit(X, y, epochs=early_stopping.patience, batch_size=32, verbose=0)

    # Save scaler
    with open("scaler_params.json", "w") as f:
        json.dump({
            "data_min": target_scaler.data_min_.tolist(),
            "data_scale": target_scaler.scale_.tolist()
        }, f)

    bucket = "stockmarketprediction25"
    s3 = boto3.client("s3")
    s3.upload_file("scaler_params.json", bucket, "stock_lstm/scaler_params.json")

    # ======== CORRECT MODEL EXPORT (NO SERVO) ========
    model_root = "/home/sagemaker-user/model"
    model_dir = f"{model_root}/1"

    if os.path.exists(model_root):
        shutil.rmtree(model_root)

    os.makedirs(model_dir, exist_ok=True)

    model.export(model_dir)

    # Create tar.gz (Python-safe)
    subprocess.run(
        ["tar", "-czvf", "model.tar.gz", "-C", model_root, "."],
        check=True
    )

    s3.upload_file("model.tar.gz", bucket, "stock_lstm/model.tar.gz")

    # ======== FUTURE FORECAST (FIXED LOGIC) ========
    seq = X[-1].copy()
    preds = []

    for _ in range(future_days):
        next_scaled = model.predict(seq[np.newaxis, ...], verbose=0)[0]
        next_values = target_scaler.inverse_transform([next_scaled])[0]
        preds.append(next_values)

        next_row = seq[-1].copy()
        next_row[features.index("Open")] = next_scaled[targets.index("Open")]
        next_row[features.index("Close")] = next_scaled[targets.index("Close")]

        seq = np.vstack([seq[1:], next_row])

    return np.array(preds)

# -------------------------------
# 7) Main Execution
# -------------------------------
df, stock_name = get_stock_data("INFY", "2021-01-01")
if df is not None:
    df = add_indicators(df)
    csv_file_basic = f"{stock_name}_basic.csv"
    df.to_csv(csv_file_basic, index=False)
    print(f"✅ Data saved to {csv_file_basic}")
    next_3 = train_eval_predict_lstm(df, lookback=30, future_days=1)
    print("\n📈 Next Days Predicted (Open & Close):")
    for i, (op, cp) in enumerate(next_3, 1):
        print(f"Day {i}: 🟢 Open={op:.2f}, 🔴 Close={cp:.2f}")

  # -------------------------------
# 8) Deploy Endpoint (SIMPLIFIED & WORKING)
# -------------------------------
import time
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

bucket = "stockmarketprediction25"
prefix = "stock_lstm"

role = "arn:aws:iam::143063371745:role/service-role/AmazonSageMaker-ExecutionRole-20260105T124631"

model_uri = f"s3://{bucket}/{prefix}/model.tar.gz"

tf_model = TensorFlowModel(
    model_data=model_uri,
    role=role,
    framework_version="2.12",
    sagemaker_session=sagemaker.Session()
)

# ✅ Unique endpoint name (prevents ValidationException)
endpoint_name = f"stock-lstm-endpoint-{int(time.time())}"

predictor = tf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name=endpoint_name
)

print("✅ SageMaker Endpoint deployed:", endpoint_name)


# -------------------------------
# 8) To Know the shape 
# -------------------------------

import numpy as np
from sklearn.preprocessing import MinMaxScaler

feature_cols = ["Open", "High", "Low", "Volume", "Close",
                "RSI", "MACD", "MACD_SIGNAL",
                "EMA7", "EMA14", "EMA21", "EMA50", "EMA100",
                "ATR", "BB_MID", "BB_UPPER", "BB_LOWER"]

lookback = 30

# Use your existing dataframe
data = df[feature_cols].values

# Temporary scaler (for endpoint input)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Last 30 timesteps
sample_input = scaled_data[-lookback:]
sample_input = sample_input.reshape(1, lookback, len(feature_cols))

print(sample_input.shape)


# -------------------------------
# 9) Raw Prediction 
# -------------------------------


# Call endpoint
result = predictor.predict(sample_input.tolist())
print("Raw model output:", result)

# Extract predictions correctly
pred_scaled = np.array(result["predictions"])

# -------------------------------
# 10) Real predicted values , scalar params
# -------------------------------

# Load scaler params
import json
import boto3
import numpy as np

s3 = boto3.client("s3")
bucket = "stockmarketprediction25"
key = "stock_lstm/scaler_params.json"

obj = s3.get_object(Bucket=bucket, Key=key)
scaler_params = json.loads(obj["Body"].read())

data_min = np.array(scaler_params["data_min"])
scale = np.array(scaler_params["data_scale"])

pred_real = pred_scaled / scale + data_min
print("📈 Predicted prices (Open, Close):", pred_real)

# -------------------------------
# 10)to generate “instances” automatically
# -------------------------------

import numpy as np
from sklearn.preprocessing import MinMaxScaler

feature_cols = ["Open", "High", "Low", "Volume", "Close",
                "RSI", "MACD", "MACD_SIGNAL",
                "EMA7", "EMA14", "EMA21", "EMA50", "EMA100",
                "ATR", "BB_MID", "BB_UPPER", "BB_LOWER"]

lookback = 30

data = df[feature_cols].values

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

instances = scaled[-lookback:].reshape(1, lookback, len(feature_cols)).tolist()

print(json.dumps({"instances": instances}, indent=2))


# -------------------------------
# 10) Converting instance instances into your 1×30×17 tensor 
# -------------------------------
import json

payload = {
    "instances": instances  # your 1×30×17 tensor
}

print(json.dumps(payload))

# -------------------------------
# 11) 
# -------------------------------
import json
import boto3

sm = boto3.client("sagemaker")
print(sm.list_endpoints()["Endpoints"])


# -------------------------------
# 12) Meta data created in s3 bucket 
# -------------------------------

import boto3
import json
from datetime import datetime

# Create S3 client
s3 = boto3.client("s3", region_name="ap-south-1")

# Create metadata
metadata = {
    "stock_name": "INFY",
    "last_updated": datetime.utcnow().isoformat(),
    "model_type": "LSTM"
}

# Upload to S3
s3.put_object(
    Bucket="stockmarketprediction25",
    Key="stock_lstm/stock_metadata.json",
    Body=json.dumps(metadata, indent=2),
    ContentType="application/json"
)

print("✅ Metadata file created!")
print(f"📍 Location: s3://stockmarketprediction25/stock_lstm/stock_metadata.json")
print(f"📊 Stock Name: INFY")


