# 1d 20-period Donchian Breakout with 1-week MA Trend Filter and Volume Confirmation
# Hypothesis: Donchian breakouts capture volatility expansion moves.
# Combined with 1-week MA200 trend filter to avoid counter-trend trades.
# Volume confirmation ensures breakouts have institutional participation.
# Works in both bull and bear markets by only taking trades aligned with higher timeframe trend.
# Targets 10-20 trades/year with disciplined entries to avoid overtrading.

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week MA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ma200_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    ma200_1d = align_htf_to_ltf(prices, df_1w, ma200_1w)
    
    # 20-period Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for Donchian and volume SMA
        # Skip if required data not available
        if (np.isnan(ma200_1d[i]) or 
            np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 1-week MA200 OR Donchian exit (close below 20-period low)
            if close[i] < ma200_1d[i] or close[i] < low_min[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above 1-week MA200 OR Donchian exit (close above 20-period high)
            if close[i] > ma200_1d[i] or close[i] > high_max[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above 20-period high + volume confirmation + uptrend
            if (close[i] > high_max[i] and 
                vol_confirm and 
                close[i] > ma200_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below 20-period low + volume confirmation + downtrend
            elif (close[i] < low_min[i] and 
                  vol_confirm and 
                  close[i] < ma200_1d[i]):
                position = -1
                signals[i] = -0.25
    
    return signals