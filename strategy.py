# 12h_Donchian20_Breakout_Volume_TrendFilter_V1
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d trend filter.
# Long when price breaks above 12h Donchian high(20) with volume > 1.5x 20-period average and price > 1d EMA50
# Short when price breaks below 12h Donchian low(20) with volume > 1.5x 20-period average and price < 1d EMA50
# Exit when price returns to the midpoint of the Donchian channel or reverses to opposite side.
# Designed for ~15-25 trades/year per symbol with proper risk control to survive bull/bear markets.
# Uses 12h timeframe to reduce noise and frequency, with 1d trend filter to avoid counter-trend trades.
name = "12h_Donchian20_Breakout_Volume_TrendFilter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 12h Donchian channel (20-period)
    donchian_window = 20
    high_max = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    low_min = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_window, 20, 50)  # Wait for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = high_max[i]
        lower = low_min[i]
        midpoint = donchian_mid[i]
        ema_50 = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price > upper Donchian band with volume confirmation and uptrend
            if price > upper and vol > 1.5 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower Donchian band with volume confirmation and downtrend
            elif price < lower and vol > 1.5 * vol_ma and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to midpoint or breaks below lower band (reversal)
            if price <= midpoint or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint or breaks above upper band (reversal)
            if price >= midpoint or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals