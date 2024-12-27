import pandas_ta as ta

def add_ohlc_features(data):
    """
    Add OHLC (Open, High, Low, Close) as separate features if not already included.
    """
    data['open'] = data['open']
    data['high'] = data['high']
    data['low'] = data['low']
    data['close'] = data['close']
    return data

def add_sma(data, length=9):
    data['sma'] = ta.sma(data['close'], length=length)
    return data

def add_ema(data, length=14):
    data['ema'] = ta.ema(data['close'], length=length)
    return data

def add_atr(data, length=10):
    data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=length)
    return data

def add_rsi(data, length=10):
    data['rsi'] = ta.rsi(data['close'], length=length)
    return data

def add_macd(data, fast=12, slow=26, signal=9):
    macd = ta.macd(data['close'], fast=fast, slow=slow, signal=signal)
    data['macd'] = macd['MACD_12_26_9']
    data['macd_signal'] = macd['MACDs_12_26_9']
    data['macd_hist'] = macd['MACDh_12_26_9']
    return data

def add_bollinger_bands(data, length=20, std=2):
    bbands = ta.bbands(data['close'], length=length, std=std)
    data['bb_upper'] = bbands['BBU_20_2.0']
    data['bb_middle'] = bbands['BBM_20_2.0']
    data['bb_lower'] = bbands['BBL_20_2.0']
    return data

def add_momentum(data, length=10):
    data['momentum'] = ta.mom(data['close'], length=length)
    return data

def add_volatility(data, window=20):
    data['volatility'] = data['close'].rolling(window=window).std()
    return data

def add_price_change(data):
    data['price_change'] = data['close'] - data['open']
    data['percent_change'] = (data['close'] - data['open']) / data['open'] * 100
    return data

def add_lagged_features(data, lags=5):
    for lag in range(1, lags + 1):
        data[f'close_lag_{lag}'] = data['close'].shift(lag)
    return data

def add_target(data):
    data['target_direction'] = (data['close'].shift(-1) > data['close']).astype(int)
    return data

def add_candle_patterns(data):
    data['doji'] = ta.cdl_doji(data['open'], data['high'], data['low'], data['close'])
    data['hammer'] = ta.cdl_hammer(data['open'], data['high'], data['low'], data['close'])
    data['engulfing'] = ta.cdl_engulfing(data['open'], data['high'], data['low'], data['close'])
    data['morning_star'] = ta.cdl_morningstar(data['open'], data['high'], data['low'], data['close'])
    return data

def add_price_action(data):
    data['higher_high'] = data['high'].rolling(window=5).max()
    data['lower_low'] = data['low'].rolling(window=5).min()
    data['breakout_high'] = (data['high'] > data['higher_high'].shift(1)).astype(int)
    data['breakout_low'] = (data['low'] < data['lower_low'].shift(1)).astype(int)
    return data

def add_trend_features(data):
    data['trend_up'] = (data['close'] > data['close'].rolling(window=5).mean()).astype(int)
    data['trend_down'] = (data['close'] < data['close'].rolling(window=5).mean()).astype(int)
    return data

def add_rolling_features(data):
    data['rolling_mean_3'] = data['close'].rolling(window=3).mean()
    data['rolling_mean_5'] = data['close'].rolling(window=5).mean()
    return data

def add_features(data, higher_data_resampled=None, enable_features=None):
    """
    ฟังก์ชันเพิ่มฟีเจอร์ที่เลือกไว้ใน enable_features
    """
    if enable_features is None:
        enable_features = ['sma', 'ema', 'atr', 'rsi', 'macd', 'bollinger_bands', 
                           'momentum', 'volatility', 'price_change', 'lagged', 'target', 
                           'candle_patterns', 'price_action', 'trend']

    if higher_data_resampled is not None and 'higher_close' in enable_features:
        data['higher_close'] = higher_data_resampled['close']

    if 'sma' in enable_features:
        data = add_sma(data)
    if 'ema' in enable_features:
        data = add_ema(data)
    if 'atr' in enable_features:
        data = add_atr(data)
    if 'rsi' in enable_features:
        data = add_rsi(data)
    if 'macd' in enable_features:
        data = add_macd(data)
    if 'bollinger_bands' in enable_features:
        data = add_bollinger_bands(data)
    if 'momentum' in enable_features:
        data = add_momentum(data)
    if 'volatility' in enable_features:
        data = add_volatility(data)
    if 'price_change' in enable_features:
        data = add_price_change(data)
    if 'lagged' in enable_features:
        data = add_lagged_features(data)
    if 'target' in enable_features:
        data = add_target(data)
    if 'candle_patterns' in enable_features:
        data = add_candle_patterns(data)
    if 'price_action' in enable_features:
        data = add_price_action(data)
    if 'trend' in enable_features:
        data = add_trend_features(data)
    if 'ohlc' in enable_features:
        data = add_ohlc_features(data)

    data.dropna(inplace=True)
    return data
