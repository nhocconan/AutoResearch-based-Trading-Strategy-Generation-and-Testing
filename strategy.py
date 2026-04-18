#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1S1_Breakout_With_Volume_Confirmation
Hypothesis: Weekly pivot R1/S1 breakouts on daily chart with volume spike capture institutional participation.
Works in bull markets (breakouts above R1) and bear markets (breakdowns below S1). Weekly pivot provides
more significant support/resistance than daily. Volume confirms institutional interest. Target: 15-25 trades/year.
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
    
    # Weekly pivot from previous week
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot levels: P = (H+L+C)/3, R1 = C + (H-L)*1.1/2, S1 = C - (H-L)*1.1/2
    pivot_w = (high_1w + low_1w + close_1w) / 3.0
    r1_w = close_1w + (high_1w - low_1w) * 1.1 / 2.0
    s1_w = close_1w - (high_1w - low_1w) * 1.1 / 2.0
    
    # Align to daily: previous week's levels available after weekly bar closes
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_w_aligned[i]
        s1_val = s1_w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike
            if price > r1_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume spike
            elif price < s1_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below weekly pivot
            if price < pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above weekly pivot
            if price > pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0