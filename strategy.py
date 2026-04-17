#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1_S1_Breakout_Volume_Confirm
Hypothesis: On daily chart, price breaks above weekly R1 or below weekly S1 with volume confirmation.
Weekly pivots act as strong support/resistance. Breakouts capture momentum in both bull and bear markets.
Volume filter ensures breakouts are genuine. Weekly timeframe reduces noise and overtrading.
Designed for 1d to achieve 10-30 trades/year with high win rate.
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
    
    # === Weekly data for Pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Daily volume average (20-period) for volume confirmation
    vol_avg20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: covers weekly pivot calculation
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * vol_avg20[i]
        
        if position == 0:
            # Long breakout: price closes above weekly R1 with volume
            if close[i] > r1_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            
            # Short breakout: price closes below weekly S1 with volume
            if close[i] < s1_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long when price returns to weekly pivot or opposite S1
            if close[i] <= pivot_1w[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to weekly pivot or opposite R1
            if close[i] >= pivot_1w[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_R1_S1_Breakout_Volume_Confirm"
timeframe = "1d"
leverage = 1.0