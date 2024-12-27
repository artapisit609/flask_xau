# random_forest_model.py
import matplotlib.pyplot as plt
import pandas as pd
import logging

from feature_engineering import add_features
from modeling import train_random_forest
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

        # --- Step 4: Train Random Forest ---
        logging.info("Training Random Forest...")
        selected_features = enable_features
        target_column = 'atr'
        rf_model, rf_metrics = train_random_forest(data, selected_features, target_column, config['random_forest'])
        logging.info(f"Random Forest Metrics:\n{rf_metrics}")

        # --- Step 5: Evaluate Strategy ---
        logging.info("Calculating Sharpe Ratio for Random Forest...")
        predicted_returns_rf = rf_model.predict(data[selected_features])
        sharpe_ratio_rf = calculate_sharpe_ratio(predicted_returns_rf)
        logging.info(f"Sharpe Ratio (Random Forest): {sharpe_ratio_rf}")

        # --- Step 6: Visualize Results ---
        sharpe_ratios = {
            'Random Forest': sharpe_ratio_rf
        }

        plt.bar(sharpe_ratios.keys(), sharpe_ratios.values(), color='skyblue', edgecolor='black')
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
