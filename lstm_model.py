# lstm_model.py
import matplotlib.pyplot as plt
import pandas as pd
import logging

from feature_engineering import add_features
from modeling import train_lstm, prepare_lstm_data
from utils import calculate_sharpe_ratio
from config import config

from sqlalchemy import create_engine
from datetime import datetime

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data_from_mysql(engine, query):
    """
    Fetch data from MySQL using a SQLAlchemy engine.
    """
    try:
        data = pd.read_sql(query, engine)
        data['time'] = pd.to_datetime(data['time'])
        data.set_index('time', inplace=True)
        return data
    except Exception as e:
        logging.error(f"Error fetching data from MySQL: {e}")
        return None

def main():
    try:
        # --- Step 1: Database Connection ---
        engine = create_engine('mysql+pymysql://bestfix_apisit609:Word1023-a10@bestfixrepair.com/bestfix_xauusdm')
        query = "SELECT * FROM xauusdm_h1 WHERE time BETWEEN '2014-01-01' AND '2023-12-31'"

        # --- Step 2: Fetch Data ---
        logging.info("Fetching data from MySQL...")
        data = fetch_data_from_mysql(engine, query)

        if data is None or data.empty:
            raise ValueError("Data fetching failed or returned empty dataset.")

        # --- Step 3: Add Features ---
        logging.info("Adding features...")
        enable_features = ['close', 'sma', 'ema', 'atr', 'rsi', 'volatility', 'macd']
        data = add_features(data, enable_features=enable_features)

        if data.empty:
            raise ValueError("Data is empty after feature engineering.")

        # --- Step 4: Prepare Data for LSTM ---
        logging.info("Preparing data for LSTM...")
        selected_features = enable_features
        time_steps = 30

        X, y, scaler_features, scaler_target = prepare_lstm_data(data, selected_features, time_steps)

        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("LSTM data preparation failed or returned empty dataset.")

        logging.info(f"Data prepared for LSTM: X shape {X.shape}, y shape {y.shape}")

        # --- Step 5: Train LSTM ---
        logging.info("Training LSTM for Trend Detection...")
        lstm_model, lstm_metrics = train_lstm(data, selected_features, time_steps, config['lstm'])

        if lstm_model is None:
            raise ValueError("LSTM training failed.")

        logging.info(f"LSTM Metrics: {lstm_metrics}")

        # --- Step 6: Evaluate Strategy ---
        logging.info("Calculating Sharpe Ratio for LSTM...")
        predicted_returns_lstm = lstm_model.predict(X)
        sharpe_ratio_lstm = calculate_sharpe_ratio(predicted_returns_lstm.flatten())
        logging.info(f"Sharpe Ratio (LSTM): {sharpe_ratio_lstm}")

        # --- Step 7: Visualize Results ---
        plt.bar(['LSTM'], [sharpe_ratio_lstm], color='skyblue', edgecolor='black')
        plt.title("Sharpe Ratio for LSTM", fontsize=14)
        plt.ylabel("Sharpe Ratio", fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()
