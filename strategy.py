#!/usr/bin/env python3
"""
4h_HTF_Camarilla_Pivot_S1S4_Breakout_Volume_Trend
Hypothesis: Combines 12h Camarilla pivot levels (S1/S4) with volume confirmation and 4h EMA trend filter.
Uses higher timeframe structure (12h) for signal direction to reduce false breakouts and
improve performance in both bull and bear markets. Targets 20-35 trades/year.
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
    
    # Get 12h data for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (S1, S4, R1, R4)
    pivot = np.full(len(close_12h), np.nan)
    s1 = np.full(len(close_12h), np.nan)
    s4 = np.full(len(close_12h), np.nan)
    r1 = np.full(len(close_12h), np.nan)
    r4 = np.full(len(close_12h), np.nan)
    
    for i in range(2, len(close_12h)):
        high_prev = high_12h[i-1]
        low_prev = low_12h[i-1]
        close_prev = close_12h[i-1]
        range_val = high_prev - low_prev
        
        pivot[i] = (high_prev + low_prev + close_prev) / 3.0
        s1[i] = pivot[i] - (range_val * 1.0 / 6.0)
        s4[i] = pivot[i] - (range_val * 2.0 / 3.0)
        r1[i] = pivot[i] + (range_val * 1.0 / 6.0)
        r4[i] = pivot[i] + (range_val * 2.0 / 3.0)
    
    # Get 4h data for EMA trend filter (20-period)
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = close[i] * alpha + ema20[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align 12h Camarilla levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(s1_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above S1 with volume spike and 4h uptrend
            if (close[i] > s1_aligned[i] and vol_spike[i] and 
                close[i] > ema20[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below R1 with volume spike and 4h downtrend
            elif (close[i] < r1_aligned[i] and vol_spike[i] and 
                  close[i] < ema20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below S4 or 4h trend turns down
            if (close[i] < s4_aligned[i] or close[i] < ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above R4 or 4h trend turns up
            if (close[i] > r4_aligned[i] or close[i] > ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Camarilla_Pivot_S1S4_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0