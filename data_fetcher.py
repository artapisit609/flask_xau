import MetaTrader5 as mt5
import pandas as pd

def fetch_historical_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"No candle data found for {symbol}")
        return None
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.tz_convert('Asia/Bangkok')
    data.set_index('time', inplace=True)
    data['volatility'] = data['close'].rolling(window=20).std()
    return data
