#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Trend
Hypothesis: Combines 1-day Camarilla pivot breakouts with volume confirmation and 1-week trend filter.
Breakouts above R1 or below S1 require volume spike and alignment with weekly trend direction.
Designed for low trade frequency (12-37/year) to minimize fee drag while capturing trending moves in both bull and bear markets.
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
    
    # Calculate 1-day Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels (R1, S1)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = pivot_1d + range_1d * 1.1 / 12
    s1_1d = pivot_1d - range_1d * 1.1 / 12
    
    # Align 1-day levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1-week EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    k = 2 / (34 + 1)
    for i in range(34, len(close_1w)):
        if i == 34:
            ema34_1w[i] = np.mean(close_1w[0:35])
        else:
            ema34_1w[i] = close_1w[i] * k + ema34_1w[i-1] * (1 - k)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 1.5 x 24-period average (2 days of 12h data)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike and weekly uptrend
            if (close[i] > r1_1d_aligned[i] and vol_spike[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and weekly downtrend
            elif (close[i] < s1_1d_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below pivot or weekly trend turns down
            if (close[i] < pivot_1d_aligned[i] or close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above pivot or weekly trend turns up
            if (close[i] > pivot_1d_aligned[i] or close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0