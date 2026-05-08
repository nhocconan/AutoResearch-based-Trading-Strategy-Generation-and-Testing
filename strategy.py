#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume spike confirmation.
# Long when price breaks above 1d Donchian upper band (20) AND 1w EMA40 rising AND volume > 2x 20-period average.
# Short when price breaks below 1d Donchian lower band (20) AND 1w EMA40 falling AND volume > 2x 20-period average.
# Exit when price crosses back inside the 1d Donchian channel.
# Donchian channels provide clear trend-following structure. 1w EMA40 filters higher timeframe trend.
# Volume spike (2x) confirms institutional participation. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian_20_1wEMA40_Volume"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA40 for trend filter
    ema40_1w = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # 1w EMA40 direction
    ema40_rising = np.zeros_like(ema40_1w_aligned, dtype=bool)
    ema40_falling = np.zeros_like(ema40_1w_aligned, dtype=bool)
    ema40_rising[1:] = ema40_1w_aligned[1:] > ema40_1w_aligned[:-1]
    ema40_falling[1:] = ema40_1w_aligned[1:] < ema40_1w_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Sufficient warmup for EMA40 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema40_1w_aligned[i]) or np.isnan(ema40_rising[i]) or 
            np.isnan(ema40_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, 1w EMA40 rising, volume filter
            long_cond = (close[i] > high_max[i]) and ema40_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian lower, 1w EMA40 falling, volume filter
            short_cond = (close[i] < low_min[i]) and ema40_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower band
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper band
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals