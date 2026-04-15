#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses 6h Donchian breakout in direction of weekly pivot trend (price above/below weekly pivot)
# Volume confirmation requires volume > 1.5x 20-period median
# Works in bull markets (breakouts up when above weekly pivot) and bear markets (breakouts down when below weekly pivot)
# Target: 50-150 total trades over 4 years = 12-37/year

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Shift by 1 to avoid look-ahead (use prior week's pivot)
    pivot_1w = np.roll(pivot_1w, 1)
    pivot_1w[0] = np.nan
    
    # Align weekly pivot to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            continue
        
        # Volume condition: volume > 1.5x 20-period median
        vol_median = np.median(volume[max(0, i-19):i+1])
        vol_ok = volume[i] > 1.5 * vol_median
        
        # Long entry: price breaks above Donchian high AND price above weekly pivot
        if (close[i] > highest_high[i] and 
            close[i] > pivot_1w_aligned[i] and 
            vol_ok and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low AND price below weekly pivot
        elif (close[i] < lowest_low[i] and 
              close[i] < pivot_1w_aligned[i] and 
              vol_ok and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite breakout or price crosses weekly pivot in opposite direction
        elif position == 1 and (close[i] < lowest_low[i] or close[i] < pivot_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_high[i] or close[i] > pivot_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0