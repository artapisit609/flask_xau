import MetaTrader5 as mt5
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
import pandas_ta as ta

# **1. ดึงข้อมูลจาก MT5**
def fetch_mt5_data(symbol="XAUUSDm", timeframe=mt5.TIMEFRAME_M1, num_bars=1000):
    mt5.initialize()
    data = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    mt5.shutdown()
    if data is None:
        raise ValueError("ไม่สามารถดึงข้อมูลจาก MetaTrader 5 ได้")
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# **2. Feature Engineering**
def add_features(df):
    df['EMA_10'] = ta.ema(df['close'], length=10)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df.bfill(inplace=True)  # เติมค่าที่หายไป
    return df

# **3. Scaling Data**
def scale_data(df, feature_columns):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])
    df_scaled = pd.DataFrame(scaled_data, columns=feature_columns, index=df.index)
    return df_scaled, scaler

# **4. การสร้างข้อมูลลำดับ**
def create_sequences(data, sequence_length):
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length, :-1])
        labels.append(data[i + sequence_length, -1])
    return np.array(sequences), np.array(labels)

# **5. โมเดล LSTM**
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# **6. โมเดล XGBoost**
def build_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

# **7. การบันทึกโมเดล**
def save_models(lstm_model, xgb_model, lstm_path='lstm_model.keras', xgb_path='xgb_model.json'):
    lstm_model.save(lstm_path)
    xgb_model.save_model(xgb_path)

# **8. การโหลดโมเดล**
def load_models(lstm_path='lstm_model.keras', xgb_path='xgb_model.json'):
    lstm_model = tf.keras.models.load_model(lstm_path)
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    return lstm_model, xgb_model

# **9. Pipeline**
def main():
    symbol = "XAUUSDm"
    feature_columns = ['open', 'high', 'low', 'close', 'EMA_10', 'ATR', 'RSI']
    
    # Step 1: ดึงข้อมูล
    df = fetch_mt5_data(symbol)
    
    # Step 2: เพิ่ม Features
    df = add_features(df)

    # Step 3: Scaling ข้อมูล
    df_scaled, scaler = scale_data(df, feature_columns)
    
    # Step 4: แบ่งข้อมูล
    train_size = int(len(df_scaled) * 0.7)
    val_size = int(len(df_scaled) * 0.2)
    train = df_scaled[:train_size]
    val = df_scaled[train_size:train_size + val_size]
    test = df_scaled[train_size + val_size:]
    
    # Step 5: สร้างข้อมูลลำดับ
    sequence_length = 20
    X_train, y_train = create_sequences(train.values, sequence_length)
    X_val, y_val = create_sequences(val.values, sequence_length)
    X_test, y_test = create_sequences(test.values, sequence_length)

    # Step 6: ฝึกโมเดล LSTM
    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    lstm_predictions = lstm_model.predict(X_test)

    # Step 7: ฝึกโมเดล XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    xgb_model = build_xgboost_model(X_train_flat, y_train)
    xgb_predictions = xgb_model.predict(X_test_flat)

    # Step 8: Ensemble Model
    ensemble_predictions = (lstm_predictions.flatten() + xgb_predictions) / 2

    # Step 9: Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions))
    print(f"RMSE: {rmse}")

    # Step 10: บันทึกโมเดล
    save_models(lstm_model, xgb_model)

    # Step 11: โหลดโมเดลกลับมาใช้งาน
    loaded_lstm_model, loaded_xgb_model = load_models()
    print("โมเดลถูกโหลดกลับมาเรียบร้อยแล้ว!")

if __name__ == "__main__":
    main()
