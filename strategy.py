# Hypothesis: 4h Donchian breakout with 1-day ADX trend filter and volume confirmation
# Long when price breaks above 20-period high AND ADX(14) > 25 AND volume > 1.5x average
# Short when price breaks below 20-period low AND ADX(14) > 25 AND volume > 1.5x average
# Exit when price returns to 10-period moving average
# Uses price channels for trend following, ADX to filter strong trends, volume for confirmation
# Target: 50-150 total trades over 4 years (12-38/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Calculate Donchian channels on 4h (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Calculate 10-period moving average for exit
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of 20 for Donchian + buffer)
    start = 35
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ma_10[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get ADX value aligned to 4h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
        adx_val = adx_aligned[i]
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above 20-period high AND strong trend (ADX > 25) AND volume confirmation
            if (price > highest_high[i] and adx_val > 25 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below 20-period low AND strong trend (ADX > 25) AND volume confirmation
            elif (price < lowest_low[i] and adx_val > 25 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 10-period moving average
            if price <= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 10-period moving average
            if price >= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_ADX_Volume_Trend"
timeframe = "4h"
leverage = 1.0