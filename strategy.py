#!/usr/bin/env python3
# 6h_1d_weekly_pivot_breakout
# Hypothesis: 6-hour breakout of weekly pivot support/resistance levels with volume confirmation.
# Uses weekly pivot points (calculated from prior week) as key support/resistance.
# Long when price breaks above weekly R1 with volume confirmation.
# Short when price breaks below weekly S1 with volume confirmation.
# Works in both bull and bear markets by trading breakouts of significant weekly levels.
# Volume filter ensures breakouts have participation, reducing false signals.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

name = "6h_1d_weekly_pivot_breakout"
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's data for pivot calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # Weekly pivot point calculation (standard formula)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Weekly support and resistance levels
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above weekly R1 with volume confirmation
        if (close[i] > r1_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below weekly S1 with volume confirmation
        elif (close[i] < s1_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or price returns to weekly pivot level
        elif position == 1 and close[i] < pivot[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > pivot[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals