#!/usr/bin/env python3
# 6d_donchian_breakout_weekly_pivot_volume_v1
# Hypothesis: 6h timeframe trading using Donchian channel breakouts with weekly pivot direction and volume confirmation.
# Donchian channels identify breakouts in both bull and bear markets. Weekly pivot provides higher timeframe bias.
# Volume confirmation ensures breakout validity. Target: 50-150 total trades over 4 years (12-37/year).

name = "6d_donchian_breakout_weekly_pivot_volume_v1"
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
    
    # Get weekly data for pivot points - call ONCE before loop
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points
    pivot_w = (high_w + low_w + close_w) / 3
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    
    # Get daily data for Donchian channel - call ONCE before loop
    df_d = get_htf_data(prices, '1d')
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    
    # Calculate 20-period Donchian channel on daily timeframe
    upper_d = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    lower_d = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly pivots and daily Donchian to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    upper_d_aligned = align_htf_to_ltf(prices, df_d, upper_d)
    lower_d_aligned = align_htf_to_ltf(prices, df_d, lower_d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned indicators for current 6h bar
        pivot = pivot_w_aligned[i]
        r1 = r1_w_aligned[i]
        s1 = s1_w_aligned[i]
        upper = upper_d_aligned[i]
        lower = lower_d_aligned[i]
        
        # Skip if any required data is NaN
        if (np.isnan(pivot) or np.isnan(r1) or np.isnan(s1) or 
            np.isnan(upper) or np.isnan(lower) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume breakout condition: current volume > 1.5x 20-period average
        vol_breakout = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below daily Donchian lower
            if not np.isnan(lower) and close[i] < lower:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above daily Donchian upper
            if not np.isnan(upper) and close[i] > upper:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper with volume confirmation and price above weekly pivot
            if (not np.isnan(upper) and high[i] >= upper and close[i] > upper and 
                vol_breakout and close[i] > pivot):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower with volume confirmation and price below weekly pivot
            elif (not np.isnan(lower) and low[i] <= lower and close[i] < lower and 
                  vol_breakout and close[i] < pivot):
                position = -1
                signals[i] = -0.25
    
    return signals