#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeFilter
Hypothesis: On 6h timeframe, enter long when price breaks above 20-period Donchian high 
with volume confirmation and weekly pivot direction bias; short when breaks below 
Donchian low with volume and opposite weekly pivot bias. Weekly pivot (from prior week) 
provides directional bias to avoid counter-trend trades. Designed for 50-150 total trades 
over 4 years to minimize fee drag and work in both bull/bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily data for Donchian channels (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily data
    # Upper = max(high, 20), Lower = min(low, 20)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === Weekly pivot points from prior week ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (standard calculation)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    
    # Align weekly pivot levels to 6h
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === Daily volume average for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers 20-day Donchian and volume average
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.3x 20-day average
        vol_filter = vol_1d_current > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Pivot direction bias: price above/below weekly pivot point
        above_pivot = close[i] > pp_1w_aligned[i]
        below_pivot = close[i] < pp_1w_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price > Donchian high + volume + above weekly pivot
            if close[i] > donchian_high_aligned[i] and vol_filter and above_pivot:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < Donchian low + volume + below weekly pivot
            elif close[i] < donchian_low_aligned[i] and vol_filter and below_pivot:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal at opposite Donchian level
        elif position == 1:
            if close[i] < donchian_low_aligned[i]:  # break below Donchian low = exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > donchian_high_aligned[i]:  # break above Donchian high = exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeFilter"
timeframe = "6h"
leverage = 1.0