#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Breakout_Volume_Filter_v1
Breakout above weekly pivot R1 or below weekly pivot S1 with volume confirmation.
Weekly pivot levels calculated from prior week's high/low/close.
Uses 1d timeframe for execution with weekly pivot as filter.
Volume filter: current volume > 1.5x 20-day average volume.
Exit when price returns to weekly pivot point (PP).
Designed to capture institutional breakouts with institutional volume.
Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # === Weekly Pivot Points (calculated from prior week) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate pivot points for each week: PP = (H+L+C)/3
    # Then R1 = 2*PP - L, S1 = 2*PP - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivot levels to daily timeframe (with 1-week delay for completion)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Volume Filter: Current volume > 1.5x 20-day average volume ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if (close[i] > r1_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to pivot point (PP)
        elif position == 1:
            # Exit long: price returns to or below PP
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above PP
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_Breakout_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0