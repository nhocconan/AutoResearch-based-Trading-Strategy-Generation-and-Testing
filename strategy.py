#!/usr/bin/env python3
# 6h_donchian_20_12h_pivot_direction_v1
# Hypothesis: 6h Donchian(20) breakout with 12h pivot direction filter.
# Enters long on breakout above 20-period high when 12h pivot shows bullish bias (price above pivot),
# enters short on breakdown below 20-period low when 12h pivot shows bearish bias (price below pivot).
# Uses volume confirmation (>1.5x 20-period average) to filter false breakouts.
# Designed for 60-120 total trades over 4 years (15-30/year) to avoid fee drag.
# Works in bull markets by catching uptrend continuations and in bear markets by catching downtrend continuations.

name = "6h_donchian_20_12h_pivot_direction_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume filter: 20-period average volume
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    
    # Get 12h data for pivot points
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate classic pivot points: P = (H+L+C)/3
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    
    # Align 12h pivot to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = lookback + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_12h_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low or volume fails
            if close[i] < lowest_low[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high or volume fails
            if close[i] > highest_high[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high, volume confirmation, and price above 12h pivot (bullish bias)
            if (close[i] > highest_high[i] and 
                volume_filter and 
                close[i] > pivot_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below Donchian low, volume confirmation, and price below 12h pivot (bearish bias)
            elif (close[i] < lowest_low[i] and 
                  volume_filter and 
                  close[i] < pivot_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals