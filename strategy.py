# 12h_1w_donchian_volume_v1
# Hypothesis: 12-hour Donchian channel breakout with volume confirmation and weekly trend filter.
# In bull markets, buy breakouts above weekly trend; in bear markets, sell breakdowns below weekly trend.
# Uses weekly higher timeframe to filter direction and avoid counter-trend trades.
# Target: 15-30 trades/year to minimize fee drag while capturing strong moves.

name = "12h_1w_donchian_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema = close_1w.copy()
        alpha = 2 / (20 + 1)
        for i in range(1, len(close_1w)):
            ema[i] = alpha * close_1w[i] + (1 - alpha) * ema[i-1]
        ema_20_1w = ema
    
    # Align weekly EMA to 12h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4-period average (48h)
    vol_ma_4 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 4:
            vol_sum -= volume[i-4]
        if i >= 3:
            vol_ma_4[i] = vol_sum / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR weekly trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR weekly trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above Donchian upper band with volume confirmation AND price above weekly EMA (bullish)
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if (close[i] > highest_high[i] and 
                vol_ratio > 2.0 and 
                close[i] > ema_20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian lower band with volume confirmation AND price below weekly EMA (bearish)
            elif (close[i] < lowest_low[i] and 
                  vol_ratio > 2.0 and 
                  close[i] < ema_20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals