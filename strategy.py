#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1_S1_Breakout_Volume
Hypothesis: Uses weekly pivot points (R1/S1) from weekly timeframe for breakout signals,
confirmed by volume spike on daily timeframe. Designed to capture strong directional moves
while avoiding false breakouts in ranging markets. Weekly pivots provide strong support/resistance
levels that work in both bull and bear markets, with volume confirmation ensuring momentum.
Target: 15-25 trades/year.
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
    
    # Get weekly data for pivot point calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    
    # Align weekly pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Volume spike: current volume > 2.0 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume spike
            if close[i] > r1_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike
            elif close[i] < s1_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below weekly pivot
            if close[i] < pivot[i - len(df_weekly) * 7 // 1 if i >= len(df_weekly) * 7 // 1 else 0] if hasattr(pivot, '__len__') else pivot[0]:
                # Simplified: exit when price crosses below weekly S1 (more conservative)
                if close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly pivot
            if close[i] > pivot[i - len(df_weekly) * 7 // 1 if i >= len(df_weekly) * 7 // 1 else 0] if hasattr(pivot, '__len__') else pivot[0]:
                # Simplified: exit when price crosses above weekly R1 (more conservative)
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    # Fix pivot access - use aligned pivot for exit logic
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    signals = np.zeros(n)
    position = 0
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume spike
            if close[i] > r1_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike
            elif close[i] < s1_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below weekly S1 (stop and reverse level)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly R1 (stop and reverse level)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_R1_S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0