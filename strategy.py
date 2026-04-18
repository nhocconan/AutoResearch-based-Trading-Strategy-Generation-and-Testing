#!/usr/bin/env python3
"""
6h_WeeklyPivot_R1S1_Breakout_Volume
Hypothesis: Weekly pivot levels provide robust support/resistance that work across market regimes.
Breakouts above R1 or below S1 with volume confirmation capture institutional participation.
Using 6h timeframe balances responsiveness with lower transaction costs. Weekly timeframe
avoids overfitting to daily noise while capturing multi-week trends. Volume filters ensure
breakouts have conviction. Works in both bull (breakout continuation) and bear (breakdown) markets.
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
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = np.full(len(weekly_high), np.nan)
    r1 = np.full(len(weekly_high), np.nan)
    s1 = np.full(len(weekly_high), np.nan)
    
    for i in range(len(weekly_high)):
        pivot[i] = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
        r1[i] = 2 * pivot[i] - weekly_low[i]
        s1[i] = 2 * pivot[i] - weekly_high[i]
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
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
            # Long: break above weekly R1 with volume spike
            if close[i] > r1_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike
            elif close[i] < s1_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below weekly pivot (mean reversion)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly pivot (mean reversion)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0