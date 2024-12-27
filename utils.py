import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Calculate the Sharpe Ratio for a given set of returns.
    """
    try:
        excess_returns = returns - risk_free_rate
        mean_excess_returns = np.mean(excess_returns)
        std_dev_excess_returns = np.std(excess_returns)
        if std_dev_excess_returns == 0:
            raise ValueError("Standard deviation of returns is zero, cannot calculate Sharpe Ratio.")
        sharpe_ratio = mean_excess_returns / std_dev_excess_returns
        return sharpe_ratio
    except Exception as e:
        logging.error(f"Error in calculating Sharpe Ratio: {e}")
        return None
