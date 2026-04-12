# [Experiment 37104] Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# In bull markets: buy when price breaks above 20-day high with volume, weekly EMA up
# In bear markets: sell when price breaks below 20-day low with volume, weekly EMA down
# Weekly EMA filter ensures we only trade with the higher timeframe trend
# Target: 15-25 trades/year per symbol to minimize fee drag

name = "1d_donchian_weekly_trend_volume"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 21:
        return np.zeros(n)
    
    # Weekly EMA21 for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False).values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema21)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly trend not ready
        if np.isnan(weekly_ema21_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long: price breaks above 20-day high with volume, weekly EMA rising
        if (not np.isnan(high_20[i]) and 
            close[i] > high_20[i] and 
            vol_confirm[i] and 
            weekly_ema21_aligned[i] > weekly_ema21_aligned[i-1] and  # EMA rising
            position != 1):
            position = 1
            signals[i] = 0.25
        
        # Short: price breaks below 20-day low with volume, weekly EMA falling
        elif (not np.isnan(low_20[i]) and 
              close[i] < low_20[i] and 
              vol_confirm[i] and 
              weekly_ema21_aligned[i] < weekly_ema21_aligned[i-1] and  # EMA falling
              position != -1):
            position = -1
            signals[i] = -0.25
        
        # Exit: opposite breakout or EMA flatten
        elif position == 1 and (close[i] < low_20[i] or 
                                weekly_ema21_aligned[i] <= weekly_ema21_aligned[i-1]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > high_20[i] or 
                                 weekly_ema21_aligned[i] >= weekly_ema21_aligned[i-1]):
            position = 0
            signals[i] = 0.0
        
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals