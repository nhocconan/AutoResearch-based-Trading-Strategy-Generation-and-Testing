#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction + volume confirmation
# - Long when price breaks above Donchian(20) high AND weekly pivot > prior weekly pivot (bullish bias)
# - Short when price breaks below Donchian(20) low AND weekly pivot < prior weekly pivot (bearish bias)
# - Requires volume > 1.5x 20-period average for confirmation
# - Uses weekly pivot points from prior week (no look-ahead)
# - Designed to capture momentum in trending markets while avoiding chop
# - Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (prior week's data)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Align to 6h timeframe (prior week's pivot is known after weekly close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 6h Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if NaN in indicators
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly pivot bias (using prior week's pivot)
        pivot_bullish = pivot_1w_aligned[i] > pivot_1w_aligned[i-1] if i > 0 else False
        pivot_bearish = pivot_1w_aligned[i] < pivot_1w_aligned[i-1] if i > 0 else False
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Enter long: breakout above Donchian high with bullish weekly pivot bias
            long_signal = False
            if has_volume and pivot_bullish:
                if price > highest_high[i]:
                    long_signal = True
            
            # Enter short: breakdown below Donchian low with bearish weekly pivot bias
            short_signal = False
            if has_volume and pivot_bearish:
                if price < lowest_low[i]:
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakdown below Donchian low
            if price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above Donchian high
            if price > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0