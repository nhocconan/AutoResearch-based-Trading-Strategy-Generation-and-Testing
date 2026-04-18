#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1_S1_Breakout_Volume
Hypothesis: Uses weekly Camarilla pivot levels (R1, S1) on 1d chart with volume confirmation.
Designed to work in both bull and bear markets by fading extremes at weekly pivot levels.
Target: 15-25 trades/year (~60-100 total over 4 years).
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
    
    # Get weekly data for Camarilla pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    pivot = np.full(len(high_w), np.nan)
    r1 = np.full(len(high_w), np.nan)
    s1 = np.full(len(high_w), np.nan)
    
    for i in range(len(high_w)):
        if not (np.isnan(high_w[i]) or np.isnan(low_w[i]) or np.isnan(close_w[i])):
            pivot[i] = (high_w[i] + low_w[i] + close_w[i]) / 3.0
            r1[i] = pivot[i] + (high_w[i] - low_w[i]) * 1.1 / 12.0
            s1[i] = pivot[i] - (high_w[i] - low_w[i]) * 1.1 / 12.0
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align weekly pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above S1 with volume spike (bounce from support)
            if close[i] > s1_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with volume spike (rejection at resistance)
            elif close[i] < r1_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R1 or loses momentum
            if close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S1 or loses momentum
            if close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_R1_S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0