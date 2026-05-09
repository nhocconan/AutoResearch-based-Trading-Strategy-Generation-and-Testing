# [EXPERIMENT #142479] 6h_Donchian20_12hTrend_VolumeBreakout
# Hypothesis: Use 12h Donchian breakout with volume confirmation and 12h trend filter on 6h timeframe.
# Long when price breaks above 12h Donchian upper with volume spike and bullish 12h trend.
# Short when price breaks below 12h Donchian lower with volume spike and bearish 12h trend.
# Designed to capture momentum in both bull and bear markets by following 12h trend.
# Target: 15-30 trades/year to minimize fee drag on 6h timeframe.
name = "6h_Donchian20_12hTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high over last 20 periods
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0x 20-period EMA (higher threshold for fewer trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + volume spike + bullish 12h trend
            if (price > upper_aligned[i] and vol_confirm[i] and price > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + volume spike + bearish 12h trend
            elif (price < lower_aligned[i] and vol_confirm[i] and price < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below upper Donchian or trend turns bearish
            if price < upper_aligned[i] or price < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above lower Donchian or trend turns bullish
            if price > lower_aligned[i] or price > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals