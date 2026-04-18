#!/usr/bin/env python3
"""
1d_Pivot_R1S1_Breakout_Volume_1wTrend
Hypothesis: Daily chart breakouts above weekly R1 or below weekly S1 with volume confirmation work in both bull and bear markets. Weekly trend filter ensures alignment with higher timeframe momentum. Target: 15-30 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly R1 and S1 from 1-week data
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot and R1/S1 levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1_1w = pivot_1w + range_1w * 1.1 / 12
    s1_1w = pivot_1w - range_1w * 1.1 / 12
    
    # Align weekly levels to daily timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Weekly EMA21 trend filter
    ema21_1w = np.full(len(close_1w), np.nan)
    k = 2 / (21 + 1)
    for i in range(21, len(close_1w)):
        if i == 21:
            ema21_1w[i] = np.mean(close_1w[0:22])
        else:
            ema21_1w[i] = close_1w[i] * k + ema21_1w[i-1] * (1 - k)
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume spike: current volume > 1.8 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume spike and weekly uptrend
            if (close[i] > r1_1w_aligned[i] and vol_spike[i] and 
                close[i] > ema21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike and weekly downtrend
            elif (close[i] < s1_1w_aligned[i] and vol_spike[i] and 
                  close[i] < ema21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly EMA21
            if close[i] < ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly EMA21
            if close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Pivot_R1S1_Breakout_Volume_1wTrend"
timeframe = "1d"
leverage = 1.0