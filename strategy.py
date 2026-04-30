#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above 20-day high AND price > 1w EMA34 AND volume > 2.0x 20-day average.
# Short when price breaks below 20-day low AND price < 1w EMA34 AND volume > 2.0x 20-day average.
# Exit when price crosses the 10-day midpoint (mean reversion exit).
# Donchian channels provide clear breakout levels, effective in capturing trends.
# 1w EMA34 filters for dominant weekly trend to avoid counter-trend entries.
# Volume confirmation ensures institutional participation.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_Donchian20_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels and volume MA (primary timeframe)
    # We need 20-period lookback for Donchian and volume MA
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 10-day midpoint for exit (mean of 20-day high/low)
    midpoint_10 = (high_20 + low_20) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for 20-period indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_volume_confirm = curr_volume > (2.0 * vol_ma_20[i])
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 20-day high, uptrend (price > 1w EMA34), volume confirmation
            if (curr_high > high_20[i] and  # breakout above 20-day high
                curr_close > ema_34_1w_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low, downtrend (price < 1w EMA34), volume confirmation
            elif (curr_low < low_20[i] and  # breakout below 20-day low
                  curr_close < ema_34_1w_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below 10-day midpoint (mean reversion)
            if curr_close < midpoint_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above 10-day midpoint (mean reversion)
            if curr_close > midpoint_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals