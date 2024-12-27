# xgboost_model.py
import matplotlib.pyplot as plt
import pandas as pd
import logging

from feature_engineering import add_features
from modeling import train_xgboost
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

        # --- Step 4: Train XGBoost ---
        logging.info("Training XGBoost...")
        selected_features = enable_features
        target_column = 'rsi'
        xgb_model, xgb_metrics = train_xgboost(data, selected_features, target_column, config['xgboost'])

        logging.info(f"XGBoost Metrics:\n{xgb_metrics}")

        # --- Step 5: Evaluate Strategy ---
        logging.info("Calculating Sharpe Ratio...")
        predicted_returns = xgb_model.predict(data[selected_features])
        sharpe_ratio = calculate_sharpe_ratio(predicted_returns)
        logging.info(f"Sharpe Ratio: {sharpe_ratio}")

        # --- Step 6: Visualize Results ---
        plt.bar(['XGBoost'], [sharpe_ratio], color='skyblue', edgecolor='black')
        plt.title("Sharpe Ratio Comparison", fontsize=14)
        plt.ylabel("Sharpe Ratio", fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()
