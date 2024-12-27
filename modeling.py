# modeling.py
import xgboost as xgb
import numpy as np
import logging

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

from config import config

# --- Load configurations ---
rf_config = config['random_forest']
xgb_config = config['xgboost']
lstm_config = config['lstm']

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def print_metrics(metrics):
    """
    Print model metrics in a readable format.
    """
    logging.info("Model Metrics:")
    for key, value in metrics.items():
        logging.info(f"{key}: {value:.4f}")

def check_columns(data, columns):
    """
    Check if required columns exist in the DataFrame.
    """
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")

# --- Random Forest ---
def train_random_forest(data, selected_features, target_column, config, use_randomized_search=False):
    """
    Train a Random Forest Regressor using GridSearchCV or RandomizedSearchCV.
    """
    try:
        logging.info("Starting Random Forest Training...")
        check_columns(data, selected_features + [target_column])

        # Prepare data
        X = data[selected_features]
        y = data[target_column]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Get parameter grid
        param_grid = config.get('param_grid')
        if not param_grid:
            raise ValueError("Random Forest param_grid is not defined in config.")

        # Select search method
        if use_randomized_search:
            grid_search = RandomizedSearchCV(
                RandomForestRegressor(random_state=42),
                param_distributions=param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_iter=10,
                random_state=42
            )
        else:
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                verbose=1
            )

        # Fit the model
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predictions and metrics
        y_pred = best_model.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        print_metrics(metrics)
        return best_model, metrics

    except Exception as e:
        logging.error(f"Error in Random Forest Training: {e}")
        return None, None

# --- XGBoost ---
def train_xgboost(data, selected_features, target_column, config):
    """
    Train an XGBoost Regressor.
    """
    try:
        logging.info("Starting XGBoost Training...")
        check_columns(data, selected_features + [target_column])

        X = data[selected_features]
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005, random_state=42)

        param_grid = config.get('param_grid')
        if not param_grid:
            raise ValueError("XGBoost param_grid is not defined in config.")

        grid_search = GridSearchCV(
            xgb.XGBRegressor(random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        print_metrics(metrics)
        return best_model, metrics
    except Exception as e:
        logging.error(f"Error in XGBoost Training: {e}")
        return None, None

# --- LSTM ---
def prepare_lstm_data(data, selected_features, time_steps, scaler_type='MinMax'):
    """
    Prepare data for LSTM Model.
    """
    try:
        check_columns(data, selected_features)

        features = data[selected_features].values
        target = data['macd'].values.reshape(-1, 1)

        scaler_features = MinMaxScaler() if scaler_type == 'MinMax' else StandardScaler()
        scaler_target = MinMaxScaler() if scaler_type == 'MinMax' else StandardScaler()

        features_normalized = scaler_features.fit_transform(features)
        target_normalized = scaler_target.fit_transform(target)

        X, y = [], []
        for i in range(len(features_normalized) - time_steps):
            X.append(features_normalized[i:i + time_steps])
            y.append(target_normalized[i + time_steps])

        return np.array(X), np.array(y), scaler_features, scaler_target
    except Exception as e:
        logging.error(f"Error in LSTM Data Preparation: {e}")
        return None, None, None, None

def build_lstm_model(input_shape, config):
    """
    Build an LSTM Model.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(config.get('lstm_units_1', 64), return_sequences=True, activation='tanh'),
        Dropout(config.get('dropout', 0.2)),
        LSTM(config.get('lstm_units_2', 64), return_sequences=False, activation='tanh'),
        Dropout(config.get('dropout', 0.1)),
        Dense(config.get('dense_units', 64), activation='relu'),
        Dense(config.get('dense_units_2', 32), activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(data, selected_features, time_steps=30, config=None):
    """
    Train an LSTM Model.
    """
    try:
        config = config or {}
        logging.info("Starting LSTM Training...")

        # Prepare data
        X, y, scaler_features, scaler_target = prepare_lstm_data(data, selected_features, time_steps)
        if X is None or y is None:
            raise ValueError("Failed to prepare LSTM data.")

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train model
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]), config)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=config.get('epochs', 60),
            batch_size=config.get('batch_size', 32),
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate model
        y_pred = model.predict(X_test)
        y_test_rescaled = scaler_target.inverse_transform(y_test)
        y_pred_rescaled = scaler_target.inverse_transform(y_pred)

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled)),
            'mae': mean_absolute_error(y_test_rescaled, y_pred_rescaled)
        }
        print_metrics(metrics)
        return model, metrics
    except Exception as e:
        logging.error(f"Error in LSTM Training: {e}")
        return None, None

# --- Ensemble ---
def ensemble_predictions(data, rf_model, xgb_model, lstm_model, selected_features, time_steps=30):
    """
    Combine predictions from RF, XGBoost, and LSTM into an ensemble.
    """
    try:
        check_columns(data, selected_features + ['close'])

        X = data[selected_features]
        y = data['close']

        rf_preds = rf_model.predict(X)
        xgb_preds = xgb_model.predict(X.values)

        X_lstm, _, _, _ = prepare_lstm_data(data, selected_features, time_steps)
        lstm_preds = lstm_model.predict(X_lstm)

        ensemble_preds = (0.4 * rf_preds + 0.4 * xgb_preds + 0.2 * lstm_preds.flatten())

        rmse = np.sqrt(mean_squared_error(y[-len(ensemble_preds):], ensemble_preds))
        mae = mean_absolute_error(y[-len(ensemble_preds):], ensemble_preds)

        metrics = {'rmse': rmse, 'mae': mae}
        logging.info(f"Ensemble Metrics: {metrics}")
        return ensemble_preds, metrics
    except Exception as e:
        logging.error(f"Error in Ensemble Predictions: {e}")
        return None, None
