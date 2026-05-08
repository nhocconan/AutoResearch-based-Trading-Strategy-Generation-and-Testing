# US equities strategies often fail in crypto; need different approaches
# Hypothesis: 1d Bollinger Band squeeze breakout with 1w EMA34 trend filter and volume confirmation
# Designed for 25-35 trades/year with proper risk control via trend failure
# Long: price breaks above upper BB + price > 1w EMA34 + volume spike
# Short: price breaks below lower BB + price < 1w EMA34 + volume spike
# Exit: trend failure (price crosses 1w EMA34) or opposite breakout
# Bollinger squeeze reduces false breakouts, EMA34 on weekly filters trend, volume confirms breakout strength

name = "1d_Bollinger_Squeeze_Breakout_1wEMA34_VolumeFilter"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Bollinger Bands (20, 2) on 1d data
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Calculate Bollinger Band width for squeeze detection
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < 0.8 * bb_width_ma  # Bollinger squeeze
    
    # Calculate 20-day average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 2.0x 20-day average
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Look for breakout with trend and volume confirmation during squeeze
            # Long: price breaks above upper BB + uptrend + volume squeeze + volume spike
            if close[i] > bb_upper[i] and squeeze_condition[i] and ema34_1w_aligned[i] > 0:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below lower BB + downtrend + volume squeeze + volume spike
            elif close[i] < bb_lower[i] and squeeze_condition[i] and ema34_1w_aligned[i] < 0:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: trend failure (price crosses below EMA34) or opposite breakout
            if ema34_1w_aligned[i] <= 0 or close[i] < bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend failure (price crosses above EMA34) or opposite breakout
            if ema34_1w_aligned[i] >= 0 or close[i] > bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals