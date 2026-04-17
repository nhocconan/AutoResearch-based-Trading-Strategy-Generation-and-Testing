#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1_S1_Breakout_Volume_Confirm
Hypothesis: On daily timeframe, enter long when price breaks above weekly pivot R1 with volume confirmation; short when breaks below S1. Uses weekly pivot levels as institutional support/resistance, volume to confirm institutional participation, and avoids low-volume breakouts. Designed for 10-25 trades/year to minimize fee drift and work in both bull/bear regimes via mean-reversion at extreme levels.
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
    
    # === Weekly data for pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Point and support/resistance levels
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L
    r1_1w = 2 * pivot_1w - low_1w
    # S1 = 2*P - H
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly levels to daily timeframe (delayed by 1 week for completed bar)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume confirmation: 20-day average volume
    vol_avg20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: covers weekly pivot calculation and volume average
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(vol_avg20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        vol_filter = volume[i] > 1.5 * vol_avg20[i]
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above R1 with volume
            if close[i] > r1_1w_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume
            elif close[i] < s1_1w_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse when price returns to pivot level
        elif position == 1:
            if close[i] < pivot_1w_aligned[i]:  # exit long when price returns to pivot
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > pivot_1w_aligned[i]:  # exit short when price returns to pivot
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_R1_S1_Breakout_Volume_Confirm"
timeframe = "1d"
leverage = 1.0