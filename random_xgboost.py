import pandas as pd
import logging
from sqlalchemy import create_engine
from datetime import datetime
from feature_engineering import add_features, add_lagged_features
from modeling import train_random_forest, train_xgboost
from utils import calculate_sharpe_ratio
from config import config
from sklearn.model_selection import TimeSeriesSplit

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data_from_mysql(query, connection_string):
    """
    Fetch data from MySQL using a SQL Query
    """
    try:
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            data = pd.read_sql(query, connection)
        return data
        
        import logging

    except Exception as e:
        logging.error(f"Error fetching data from MySQL: {e}")
        raise

def ensemble_predictions(data, rf_model, xgb_model, selected_features):
    """
    Combine predictions from RF and XGBoost into an ensemble.
    """
    try:
        # Prepare data for prediction
        X = data[selected_features]

        # Random Forest Predictions
        rf_preds = rf_model.predict(X.values)

        # XGBoost Predictions
        xgb_preds = xgb_model.predict(X.values)

        # Weighted Ensemble
        ensemble_preds = (0.5 * rf_preds + 0.5 * xgb_preds)

        return ensemble_preds
    except Exception as e:
        logging.error(f"Error in Ensemble Predictions: {e}")
        return None

def main():
    try:
        # --- Step 1: Connect to MySQL and Fetch Data ---
        logging.info("Connecting to MySQL and fetching data...")
        connection_string = 'mysql+pymysql://bestfix_apisit609:Word1023-a10@bestfixrepair.com/bestfix_xauusdm'
        query = "SELECT * FROM xauusdm_h1 WHERE time BETWEEN '2014-01-01' AND '2023-12-31'"
        data = fetch_data_from_mysql(query, connection_string)
        if data.empty:
            raise ValueError("No data retrieved from the database.")

        # --- Step 2: Preprocess Data ---
        logging.info("Preprocessing and adding features...")
        enable_features = ['close', 'sma', 'ema', 'atr', 'rsi', 'volatility', 'macd']
        data = add_features(data, enable_features=enable_features)
        data = add_lagged_features(data, lags=5)
        data['time'] = pd.to_datetime(data['time'])
        data.set_index('time', inplace=True)

        # Create Target Columns for Entry Signal, TP, and SL
        data['entry_signal'] = data['rsi']
        data['tp_sl'] = data['atr']
        data.dropna(inplace=True)

        # --- Step 3: Train Models ---
        selected_features = ['rsi', 'atr', 'close_lag_1', 'close_lag_2']

        # Split data using TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)

        # Train XGBoost for Entry Signal
        logging.info("Training XGBoost for Entry Signal...")
        xgb_entry_model, xgb_entry_metrics = train_xgboost(data, selected_features, 'rsi', config['xgboost'])
        logging.info(f"XGBoost Entry Signal Metrics: {xgb_entry_metrics}")

        # Train Random Forest for TP and SL
        logging.info("Training Random Forest for TP and SL...")
        rf_tp_sl_model, rf_tp_sl_metrics = train_random_forest(data, selected_features, 'tp_sl', config['random_forest'], tscv)
        logging.info(f"Random Forest TP/SL Metrics: {rf_tp_sl_metrics}")

        # --- Step 4: Predict and Evaluate ---
        logging.info("Making predictions...")
        new_data = data.iloc[-30:]

        # Ensure no missing values in new_data
        if new_data[selected_features].isnull().values.any():
            raise ValueError("New data contains NaN values in selected features.")

        # Ensemble Predictions
        ensemble_preds = ensemble_predictions(new_data, rf_tp_sl_model, xgb_entry_model, selected_features)
        if ensemble_preds is None:
            raise ValueError("Failed to generate ensemble predictions.")

        # Final Signal Evaluation
        entry_signal = ensemble_preds[-1]
        if entry_signal > 0:
            entry_signal = 1
        else:
            entry_signal = -1

        logging.info(f"Mapped Entry Signal: {entry_signal}")

        if entry_signal in [1, -1]:
            predicted_tp_sl = ensemble_preds[-1]
            predicted_tp = new_data['close'].values[-1] + (2 * predicted_tp_sl)
            predicted_sl = new_data['close'].values[-1] - (1 * predicted_tp_sl)

            # Calculate Sharpe Ratio based on predicted returns
            predicted_returns = (predicted_tp - new_data['close'].values[-1]) / new_data['close'].values[-1]
            sharpe_ratio = calculate_sharpe_ratio(predicted_returns)

            logging.info(f"Entry Signal: {entry_signal}, Predicted TP: {predicted_tp}, Predicted SL: {predicted_sl}, Sharpe Ratio: {sharpe_ratio}")
        else:
            logging.warning(f"No valid entry signal detected. Entry Signal: {entry_signal}")

    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()
