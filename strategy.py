#!/usr/bin/env python3
"""
6h_WeeklyPivot_MomentumBreakout_v1
Concept: 6h momentum breakout with weekly pivot level confirmation.
- Long: Price breaks above weekly R1 with 6h close > weekly pivot AND volume > 1.5x average
- Short: Price breaks below weekly S1 with 6h close < weekly pivot AND volume > 1.5x average
- Exit: Price crosses back below/above weekly pivot
- Uses weekly pivot from 1w data as structural reference
- Works in bull/bear: breakouts capture momentum, pivot provides mean reversion in ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_MomentumBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # === Weekly: Calculate pivot points (standard formula) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L
    r1 = 2 * pp - low_1w
    # S1 = 2*P - H
    s1 = 2 * pp - high_1w
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === 6h: Volume filter ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Skip if any value is NaN
        if (np.isnan(pp_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(vol_ma_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume confirmation
            if prices['close'][i] > r1_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume confirmation
            elif prices['close'][i] < s1_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below weekly pivot
            if prices['close'][i] < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above weekly pivot
            if prices['close'][i] > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals