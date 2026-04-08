#!/usr/bin/env python3
"""
6H Donchian Breakout + Weekly Pivot + Volume Filter
Hypothesis: Donchian(20) breakouts on 6h, filtered by weekly pivot direction (price above/below weekly pivot),
and volume confirmation (>1.5x 20-period average) capture strong momentum. Weekly pivot provides
longer-term bias to avoid counter-trend trades. Designed for 6h to balance trade frequency and signal
quality in both bull and bear markets. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot (using prior week's OHLC)
    prev_week_high = df_1w['high'].shift(1)
    prev_week_low = df_1w['low'].shift(1)
    prev_week_close = df_1w['close'].shift(1)
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    
    # Donchian channels (20-period) on 6h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(weekly_pivot_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or weekly pivot
            if close[i] <= low_min[i] or close[i] < weekly_pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or weekly pivot
            if close[i] >= high_max[i] or close[i] > weekly_pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above Donchian high with weekly pivot support and volume
            if (close[i] >= high_max[i] and 
                close[i] > weekly_pivot_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short below Donchian low with weekly pivot resistance and volume
            elif (close[i] <= low_min[i] and 
                  close[i] < weekly_pivot_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals