# 12h_1wDonchian20_1dVolumeSpike
# Hypothesis: Weekly Donchian breakout with daily volume confirmation. Long when price breaks above weekly Donchian high with volume spike, short when breaks below weekly Donchian low with volume spike. Exit when price crosses weekly Donchian median. Works in bull (breakouts) and bear (breakdowns). Target: 15-30 trades/year per symbol.
name = "12h_1wDonchian20_1dVolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    highest_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    median = (highest_high + lowest_low) / 2.0
    
    # Align to 12h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, highest_high)
    donchian_low = align_htf_to_ltf(prices, df_1w, lowest_low)
    donchian_median = align_htf_to_ltf(prices, df_1w, median)
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_median[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high with volume spike
            if price > donchian_high[i] and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian low with volume spike
            elif price < donchian_low[i] and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly Donchian median
            if price < donchian_median[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly Donchian median
            if price > donchian_median[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals