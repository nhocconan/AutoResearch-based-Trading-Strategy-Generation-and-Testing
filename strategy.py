#!/usr/bin/env python3
"""
6h_WeeklyPivot_Reversal_v1
Hypothesis: Reversals at weekly pivot levels on 6h timeframe with 1-day volume confirmation.
Uses weekly Camarilla pivot levels (R4/S4) from 1w timeframe for structure and 1d volume spike for confirmation.
Works in both bull and bear markets by fading extremes and catching reversals at key levels.
Targets 15-25 trades/year to minimize fee drag.
"""

name = "6h_WeeklyPivot_Reversal_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly Camarilla pivot levels (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous week
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Pivot point and ranges
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    
    # Camarilla levels: R4 = close + range * 1.1/2, S4 = close - range * 1.1/2
    r4_w = close_w + range_w * 1.1 / 2
    s4_w = close_w - range_w * 1.1 / 2
    
    # Align weekly levels to 6h timeframe (1 week = 28 * 6h bars)
    r4_w_aligned = align_htf_to_ltf(prices, df_1w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # 1-day volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(r4_w_aligned[i]) or np.isnan(s4_w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long reversal: price at or below S4 with volume confirmation
            if close[i] <= s4_w_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # Short reversal: price at or above R4 with volume confirmation
            elif close[i] >= r4_w_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price moves back above S4 (toward pivot) or reaches R4
            if close[i] > s4_w_aligned[i] or close[i] >= r4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price moves back below R4 (toward pivot) or reaches S4
            if close[i] < r4_w_aligned[i] or close[i] <= s4_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals