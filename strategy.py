#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian(20) breakout with 1d pivot filter and volume confirmation
# Hypothesis: Donchian breakouts capture momentum; 1d pivot provides directional bias (long above pivot, short below); volume filters false breakouts.
# Works in bull via upward breaks above pivot, in bear via downward breaks below pivot. Pivot adapts to daily market structure.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "6h_donchian20_1d_pivot_volume_v1"
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
    
    # Get daily data for pivot and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard: P = (H+L+C)/3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Calculate daily 20-period volume moving average
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d data to 6h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 20-period low OR below pivot (trend failure)
            if close[i] < lowest_20[i] or close[i] < pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above 20-period high OR above pivot (trend failure)
            if close[i] > highest_20[i] or close[i] > pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above 20-period high AND above pivot + volume confirmation
            if close[i] > highest_20[i] and close[i] > pivot_1d_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 20-period low AND below pivot + volume confirmation
            elif close[i] < lowest_20[i] and close[i] < pivot_1d_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals