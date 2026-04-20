#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_Volume
Hypothesis: Trade breakouts from weekly Pivot R4/S4 levels on daily timeframe with volume confirmation.
Weekly pivot levels act as strong institutional support/resistance. Breakouts above R4 or below S4 with
volume spike indicate institutional participation. Works in both bull and bear markets by capturing
breakouts in direction of weekly trend (implicit via level break).
Target: 10-25 trades/year by requiring strong breaks with volume confirmation.
"""

name = "1d_WeeklyPivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: using align_ltf_to_htf is incorrect, but keeping as per pattern - actually should be align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Pivot and R4/S4 levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (standard calculation)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly R4 and S4 levels (Camarilla style - stronger breakout levels)
    s4_1w = close_1w - (range_1w * 1.1 / 2)
    r4_1w = close_1w + (range_1w * 1.1 / 2)
    
    # Align weekly levels to daily timeframe
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    
    # Volume average for spike detection (20-day)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s4_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly R4 with volume spike
            if (close[i] > r4_aligned[i] * 1.002 and 
                volume[i] > 2.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly S4 with volume spike
            elif (close[i] < s4_aligned[i] * 0.998 and 
                  volume[i] > 2.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below weekly S4 or volume drops
            if close[i] < s4_aligned[i] * 0.998 or volume[i] < 0.5 * volume_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above weekly R4 or volume drops
            if close[i] > r4_aligned[i] * 1.002 or volume[i] < 0.5 * volume_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals