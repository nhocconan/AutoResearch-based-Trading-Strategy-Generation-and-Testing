#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction filter and volume confirmation
# Uses 6h Donchian(20) breakouts filtered by weekly pivot direction (above/below weekly pivot)
# and confirmed by volume spikes. Works in both bull and bear markets by taking breakouts
# in the direction of weekly bias. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe (no additional delay needed for pivot)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) on daily: 20-period high/low
    high_20d = np.full_like(high_1d, np.nan)
    low_20d = np.full_like(low_1d, np.nan)
    for i in range(20, len(high_1d)):
        high_20d[i] = np.max(high_1d[i-20:i])
        low_20d[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 6h timeframe
    high_20d_aligned = align_htf_to_ltf(prices, df_1d, high_20d)
    low_20d_aligned = align_htf_to_ltf(prices, df_1d, low_20d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(high_20d_aligned[i]) or 
            np.isnan(low_20d_aligned[i])):
            continue
        
        # Long entry: price breaks above 20-day high AND above weekly pivot + volume confirmation
        if (close[i] > high_20d_aligned[i] and 
            close[i] > pp_1w_aligned[i] and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below 20-day low AND below weekly pivot + volume confirmation
        elif (close[i] < low_20d_aligned[i] and 
              close[i] < pp_1w_aligned[i] and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite breakout or price crosses back through weekly pivot
        elif position == 1 and (close[i] < low_20d_aligned[i] or close[i] < pp_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > high_20d_aligned[i] or close[i] > pp_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0