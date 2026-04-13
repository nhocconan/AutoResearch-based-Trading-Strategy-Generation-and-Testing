# For 1d timeframe, test with weekly trend filter
# Hypothesis: Weekly Donchian breakout with daily volume confirmation
# Uses weekly price channels to capture long-term trends and daily volume to filter false breakouts
# Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-week Donchian channel on weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume and its 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.30  # 30% position size
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current daily volume > 1.5x daily volume MA
        volume_condition = volume[i] > (volume_ma_20_1d_aligned[i] * 1.5)
        
        # Entry conditions: Weekly Donchian breakout with volume confirmation
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        
        if position == 0:
            if breakout_long and volume_condition:
                position = 1
                signals[i] = position_size
            elif breakout_short and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price breaks below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price breaks above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Breakout_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0