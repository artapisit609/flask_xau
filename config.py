# config.py

config = {
    'random_forest': {
        'param_grid': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    },
    'xgboost': {
        'param_grid': {
            'learning_rate': [0.1, 0.2, 0.3],
            'n_estimators': [200, 300, 400],
            'max_depth': [5, 7, 10]
        }
    },
    'lstm': {
        'time_steps': 40,
        'epochs': 10,
        'batch_size': 32,
        'dropout': 0.1,
        'lstm_units_1': 128,
        'lstm_units_2': 64,
        'dense_units': 64,
        'dense_units_2': 32
    }
}
