# This strategy uses 1d primary timeframe with 1h trend filter
# Long when: price above 50-period EMA (1h) + price breaks above daily high of previous 3 days + volume > 1.5x 20-day avg
# Short when: price below 50-period EMA (1h) + price breaks below daily low of previous 3 days + volume > 1.5x 20-day avg
# Exit when: price crosses back below/above the 50-period EMA (1h)
# Designed for low trade frequency with trend-following edge in both bull and bear markets

name = "1d_1h_trend_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1h close
    close_1h = df_1h['close'].values
    ema_50 = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1h, ema_50)
    
    # Calculate 3-day high/low for breakout levels
    high_3d = pd.Series(high).rolling(window=3, min_periods=3).max().values
    low_3d = pd.Series(low).rolling(window=3, min_periods=3).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(high_3d[i]) or np.isnan(low_3d[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions: price above 1h EMA50 + breaks above 3-day high + volume confirmation
        if close[i] > ema_50_aligned[i] and close[i] > high_3d[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short conditions: price below 1h EMA50 + breaks below 3-day low + volume confirmation
        elif close[i] < ema_50_aligned[i] and close[i] < low_3d[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit when price crosses back below/above 1h EMA50
        elif position == 1 and close[i] < ema_50_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_50_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals