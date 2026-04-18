#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Trend
Hypothesis: 12-hour breakouts above R1 or below S1 of daily/weekly Camarilla pivots with volume confirmation and weekly EMA trend filter.
Designed for low-frequency trading (12-37 trades/year) to minimize fee drag while capturing major trend moves in both bull and bear markets.
Uses weekly EMA for trend filter to avoid whipsaws and focus on strong directional moves.
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
    
    # Calculate 1-week and 1-day Camarilla pivot levels (using weekly for higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and Camarilla levels (R1, S1)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1_1w = pivot_1w + range_1w * 1.1 / 12
    s1_1w = pivot_1w - range_1w * 1.1 / 12
    
    # Align weekly levels to 12h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Weekly EMA50 trend filter
    ema50_1w = np.full(len(close_1w), np.nan)
    k = 2 / (50 + 1)
    for i in range(50, len(close_1w)):
        if i == 50:
            ema50_1w[i] = np.mean(close_1w[0:51])
        else:
            ema50_1w[i] = close_1w[i] * k + ema50_1w[i-1] * (1 - k)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: current volume > 2.0 x 50-period average (moderate threshold for 12h)
    vol_ma = np.full(n, np.nan)
    for i in range(50, n):
        vol_ma[i] = np.mean(volume[i-50:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike and weekly uptrend
            if (close[i] > r1_1w_aligned[i] and vol_spike[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and weekly downtrend
            elif (close[i] < s1_1w_aligned[i] and vol_spike[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly pivot or weekly trend turns down
            if (close[i] < pivot_1w_aligned[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly pivot or weekly trend turns up
            if (close[i] > pivot_1w_aligned[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0