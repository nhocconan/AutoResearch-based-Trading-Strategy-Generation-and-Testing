#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA200 trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high AND 4h volume > 1.5x 20-period average AND price > 1d EMA200.
# Short when price breaks below 20-period Donchian low AND 4h volume > 1.5x 20-period average AND price < 1d EMA200.
# Exit when price crosses below 10-period Donchian low (for long) or above 10-period Donchian high (for short).
# Uses Donchian channels for trend following with trend filter to avoid counter-trend trades and volume to confirm breakouts.
# Target: 75-200 total trades over 4 years (19-50/year) for low fee drift.

name = "4h_Donchian20_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(highest_high_10[i]) or np.isnan(lowest_low_10[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-period Donchian high, volume spike, above 1d EMA200
            long_cond = (close[i] > highest_high[i]) and volume_filter[i] and (close[i] > ema200_1d_aligned[i])
            # Short conditions: price breaks below 20-period Donchian low, volume spike, below 1d EMA200
            short_cond = (close[i] < lowest_low[i]) and volume_filter[i] and (close[i] < ema200_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 10-period Donchian low
            if close[i] < lowest_low_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 10-period Donchian high
            if close[i] > highest_high_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals