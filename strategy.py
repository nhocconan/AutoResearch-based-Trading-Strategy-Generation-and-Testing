#!/usr/bin/env python3
"""
6h Weekly Pivot Direction + Volume Confirmation
Hypothesis: In BTC/ETH, weekly pivot levels act as strong support/resistance.
When price breaks above weekly R1 with volume confirmation and is above weekly pivot,
we go long. When price breaks below weekly S1 with volume confirmation and is below
weekly pivot, we go short. Uses weekly timeframe for directional bias and 6s for entry.
Designed to work in both bull (breakouts) and bear (rejections at resistance) markets.
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
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate Weekly Pivot Points (Standard)
    # Pivot = (High + Low + Close) / 3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Range = High - Low
    range_w = high_w - low_w
    # R1 = (2 * Pivot) - Low
    r1_w = (2 * pivot_w) - low_w
    # S1 = (2 * Pivot) - High
    s1_w = (2 * pivot_w) - high_w
    
    # Align weekly levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pivot_level = pivot_w_aligned[i]
        r1_level = r1_w_aligned[i]
        s1_level = s1_w_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike and above weekly pivot
            if (price > r1_level and 
                volume_spike[i] and 
                price > pivot_level):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume spike and below weekly pivot
            elif (price < s1_level and 
                  volume_spike[i] and 
                  price < pivot_level):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit conditions: price falls back below weekly pivot
            if price < pivot_level:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit conditions: price rises back above weekly pivot
            if price > pivot_level:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0