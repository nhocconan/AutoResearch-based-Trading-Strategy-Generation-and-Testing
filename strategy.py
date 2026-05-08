#2025-06-20
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and weekly trend filter
# Long when price breaks above Donchian(20) high + volume > 1.5x average + weekly uptrend
# Short when price breaks below Donchian(20) low + volume > 1.5x average + weekly downtrend
# Exit when price crosses opposite Donchian band or weekly trend reverses
# Uses 12h timeframe to target 50-150 total trades over 4 years (12-37/year)

name = "12h_Donchian20_Volume_WeeklyTrend"
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
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high + volume spike + weekly uptrend
            if (close[i] > high_20[i] and 
                volume[i] > 1.5 * vol_avg[i] and 
                ema34_1w_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + volume spike + weekly downtrend
            elif (close[i] < low_20[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  ema34_1w_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low or weekly trend turns down
            if close[i] < low_20[i] or ema34_1w_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high or weekly trend turns up
            if close[i] > high_20[i] or ema34_1w_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals