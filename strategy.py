#!/usr/bin/env python3
"""
12h_1D_Camarilla_Pivot_Breakout_Volume
Hypothesis: Uses 1d Camarilla pivot levels (R1/S1) for breakout signals on 12h timeframe.
Volume confirmation and R1/S1 bounce for exits. Targets 15-30 trades/year.
Designed to work in both bull and bear markets by fading false breakouts at key levels.
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (R1, S1)
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = np.full(len(high_1d), np.nan)
    r1 = np.full(len(high_1d), np.nan)
    s1 = np.full(len(high_1d), np.nan)
    
    for i in range(len(high_1d)):
        pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        r1[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 12.0
        s1[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 12.0
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align 1d levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume
            if close[i] > r1_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif close[i] < s1_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close back below R1 (fade the breakout) or volume drops
            if close[i] < r1_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close back above S1 (fade the breakdown) or volume drops
            if close[i] > s1_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0