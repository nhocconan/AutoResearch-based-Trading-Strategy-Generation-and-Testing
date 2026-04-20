#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction filter
# - Entry on 6h Donchian(20) breakout in direction of weekly pivot bias
# - Weekly pivot bias: price above weekly pivot = long bias, below = short bias
# - Volume confirmation: current volume > 1.5x 20-period average
# - Designed to capture institutional flow aligned with weekly structure
# - Target: 15-35 trades per year per symbol (60-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support 1 = (2 * Pivot) - High
    s1_1w = (2 * pivot_1w) - high_1w
    # Resistance 1 = (2 * Pivot) - Low
    r1_1w = (2 * pivot_1w) - low_1w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Calculate Donchian channels on 6h (20-period)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_6h = prices['volume'].values
    vol_avg_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_condition = volume_6h > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(volume_condition[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        pivot = pivot_1w_aligned[i]
        vol_ok = volume_condition[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND above weekly pivot AND volume confirmation
            if price > highest_high_20[i] and price > pivot and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND below weekly pivot AND volume confirmation
            elif price < lowest_low_20[i] and price < pivot and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR loses weekly pivot support
            if price < lowest_low_20[i] or price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR loses weekly pivot resistance
            if price > highest_high_20[i] or price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_DirectionFilter_Volume"
timeframe = "6h"
leverage = 1.0