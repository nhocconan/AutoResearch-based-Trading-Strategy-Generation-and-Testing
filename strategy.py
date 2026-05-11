#!/usr/bin/env python3
name = "6h_RangeFinder_3Bar_Range"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy import stats

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 3-bar range (highest high - lowest low)
    range_3bar = np.full(n, np.nan)
    for i in range(n):
        if i < 2:
            if i == 0:
                range_3bar[i] = high[i] - low[i]
            else:
                range_3bar[i] = max(high[i-1:i+1]) - min(low[i-1:i+1])
        else:
            range_3bar[i] = max(high[i-2:i+1]) - min(low[i-2:i+1])
    
    # 3-bar range SMA (20-period)
    range_sma = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            if i > 0:
                range_sma[i] = np.mean(range_3bar[:i+1])
        else:
            range_sma[i] = np.mean(range_3bar[i-19:i+1])
    
    # Range compression: current 3-bar range < 50% of 20-period average
    range_compressed = np.full(n, False)
    for i in range(n):
        if not np.isnan(range_3bar[i]) and not np.isnan(range_sma[i]):
            range_compressed[i] = range_3bar[i] < 0.5 * range_sma[i]
    
    # 3-bar breakout: close outside 3-bar range
    breakout_up = np.full(n, False)
    breakout_down = np.full(n, False)
    for i in range(n):
        if i >= 2:
            range_high = max(high[i-2:i+1])
            range_low = min(low[i-2:i+1])
            breakout_up[i] = close[i] > range_high
            breakout_down[i] = close[i] < range_low
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            if i > 0:
                vol_ma[i] = np.mean(volume[:i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_filter = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]):
            volume_filter[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is invalid
        if (np.isnan(range_3bar[i]) or 
            np.isnan(range_sma[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long on upward breakout from compressed range + volume
            if (range_compressed[i] and 
                breakout_up[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Enter short on downward breakout from compressed range + volume
            elif (range_compressed[i] and 
                  breakout_down[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long on breakdown or range expansion
            if (breakout_down[i] or not range_compressed[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on breakout or range expansion
            if (breakout_up[i] or not range_compressed[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals