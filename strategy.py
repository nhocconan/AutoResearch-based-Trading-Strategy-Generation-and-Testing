#!/usr/bin/env python3
"""
6h_WeeklyPivot_R1_S1_Breakout_VolumeFilter
Hypothesis: Weekly pivot R1/S1 levels act as strong support/resistance; breakouts with volume confirmation capture momentum. Works in both bull (breakouts continue) and bear (false breakouts filtered by volume).
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
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Pivot point calculation: P = (H + L + C) / 3
    pp_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    # R1 = 2*P - L
    r1_weekly = 2 * pp_weekly - low_weekly
    # S1 = 2*P - H
    s1_weekly = 2 * pp_weekly - high_weekly
    
    # Align weekly R1/S1 to 6h timeframe (aligned to previous week's close)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Volume confirmation: 20-period EMA on 6h
    volume_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 1)  # volume EMA20, need at least 2 weekly points
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ema20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period EMA
        volume_filter = volume[i] > (1.5 * volume_ema20[i])
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if close[i] > r1_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume
            elif close[i] < s1_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (mean reversion) or opposite signal
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (mean reversion) or opposite signal
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0