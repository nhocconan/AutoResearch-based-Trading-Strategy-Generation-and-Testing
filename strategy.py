#!/usr/bin/env python3
# 6h_1w_donchian_pivot_volume_v1
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Long when: price breaks above Donchian(20) high, weekly pivot > previous week pivot, and volume > 1.5x 20-period average.
# Short when: price breaks below Donchian(20) low, weekly pivot < previous week pivot, and volume > 1.5x 20-period average.
# Exit: opposite Donchian breakout or volume drops below average.
# Designed to capture strong weekly trends with institutional pivot confirmation, avoiding false breakouts in ranging markets.
# Weekly pivot provides institutional reference; Donchian breakout captures momentum; volume filter ensures conviction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_threshold = vol_ma_20 * 1.5
    
    # Weekly pivot points (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot: (high + low + close) / 3
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3
    weekly_pivot = typical_price_1w.copy()  # Simple pivot for trend direction
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Previous week pivot for trend direction
    weekly_pivot_prev = np.roll(weekly_pivot_aligned, 1)
    weekly_pivot_prev[0] = np.nan  # First value has no previous
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_threshold[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_pivot_prev[i])):
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR volume drops below average
            if price < lowest_20[i] or vol < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR volume drops below average
            if price > highest_20[i] or vol < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high, weekly pivot rising, volume confirmation
            if (price > highest_20[i] and 
                weekly_pivot_aligned[i] > weekly_pivot_prev[i] and 
                vol > volume_threshold[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low, weekly pivot falling, volume confirmation
            elif (price < lowest_20[i] and 
                  weekly_pivot_aligned[i] < weekly_pivot_prev[i] and 
                  vol > volume_threshold[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals