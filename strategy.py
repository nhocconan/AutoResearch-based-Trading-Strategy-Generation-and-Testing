#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_Volume
Hypothesis: Donchian(20) breakouts on 6h filtered by weekly pivot direction (from 1w) and volume confirmation.
Weekly pivot provides structural bias: long only when price > weekly pivot, short only when price < weekly pivot.
Reduces false breakouts in sideways markets. Targets 15-30 trades/year on 6h with low frequency to avoid fee drag.
Works in bull/bear markets by requiring volume spike and pivot directional filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points using prior week's data
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    
    # Shift by 1 to use previous week's pivot only (no look-ahead)
    pivot_1w_prev = pivot_1w.shift(1).values
    
    # Align to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w_prev)
    
    # Donchian(20) on 6h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian(20) warmup
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pivot_val = pivot_1w_aligned[i]
        upper = high_roll[i]
        lower = low_roll[i]
        
        if position == 0:
            # Long: break above Donchian upper with volume spike and price > weekly pivot (bullish bias)
            if price > upper and volume_spike[i] and price > pivot_val:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower with volume spike and price < weekly pivot (bearish bias)
            elif price < lower and volume_spike[i] and price < pivot_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to Donchian lower or weekly pivot
            if price <= lower or price <= pivot_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to Donchian upper or weekly pivot
            if price >= upper or price >= pivot_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0