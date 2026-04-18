#!/usr/bin/env python3
"""
4h_HTF_Camarilla_Pivot_S1S4_Breakout_Volume_Trend
Hypothesis: Uses daily Camarilla pivot levels (S1, S4) with volume confirmation and trend filter.
Long when price breaks above S4 with volume spike and uptrend; short when breaks below S1 with volume spike and downtrend.
Designed to work in both bull and bear markets by combining pivot-based support/resistance with volume and trend filters.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # S1 = C - (H - L) * 1.0833
    # S4 = C - (H - L) * 1.5000
    s1 = close_1d - range_1d * 1.0833
    s4 = close_1d - range_1d * 1.5000
    
    # Get 4h EMA for trend filter
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = close[i] * alpha + ema20[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Align daily S1 and S4 to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(s1_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above S4 with volume spike and 4h uptrend
            if (close[i] > s4_aligned[i] and vol_spike[i] and 
                close[i] > ema20[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and 4h downtrend
            elif (close[i] < s1_aligned[i] and vol_spike[i] and 
                  close[i] < ema20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below S1 or 4h trend turns down
            if (close[i] < s1_aligned[i] or close[i] < ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above S4 or 4h trend turns up
            if (close[i] > s4_aligned[i] or close[i] > ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Camarilla_Pivot_S1S4_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0