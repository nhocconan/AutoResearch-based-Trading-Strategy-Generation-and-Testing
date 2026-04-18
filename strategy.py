#!/usr/bin/env python3
"""
12h_1D_Camarilla_Pivot_Breakout_Volume
Hypothesis: Uses daily Camarilla pivot levels (R1, S1) for breakout signals with volume confirmation.
Designed to work in both bull and bear markets by capturing breakouts from key intraday support/resistance levels.
Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    # Formula: (H+L+C)/3 is the pivot, then ranges
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    s2 = pivot - (range_hl * 1.1 / 6)
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Volume spike: current volume > 1.5 x 24-period average (for 12h timeframe)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align daily Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 1)  # Need 24 bars for volume MA, 1 for Camarilla (already aligned)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above S1 with volume spike (bullish reversal from support)
            if close[i] > s1_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below R1 with volume spike (bearish rejection from resistance)
            elif close[i] < r1_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below S1 (support broken) or reach R1 (resistance)
            if close[i] < s1_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above R1 (resistance broken) or reach S1 (support)
            if close[i] > r1_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0