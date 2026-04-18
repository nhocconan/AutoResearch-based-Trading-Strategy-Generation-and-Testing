#!/usr/bin/env python3
"""
6h_Weekly_Pivot_R1S1_Breakout_Volume_Spike_v1
Hypothesis: Weekly Pivot R1/S1 breakouts with volume spike filter capture institutional momentum on 6B timeframe, working in both bull (breakout continuation) and bear (mean reversion at extremes) markets. Target: 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation (R1, S1)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    # Calculate weekly pivot points
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = (2 * pivot_1w) - low_1w
    s1_1w = (2 * pivot_1w) - high_1w
    
    # Align weekly levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume spike: >1.8x 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above weekly R1 with volume spike
            if price > r1 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike
            elif price < s1 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long position
            signals[i] = 0.25
            # Exit: price breaks below weekly S1 (full reversal)
            if price < s1:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Maintain short position
            signals[i] = -0.25
            # Exit: price breaks above weekly R1 (full reversal)
            if price > r1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Weekly_Pivot_R1S1_Breakout_Volume_Spike_v1"
timeframe = "6h"
leverage = 1.0