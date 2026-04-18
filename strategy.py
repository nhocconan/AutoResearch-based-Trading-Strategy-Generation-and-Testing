#!/usr/bin/env python3
"""
4h_HTF_Camarilla_Pivot_S1S4_Breakout_Volume_Trend
Hypothesis: Uses 1d Camarilla pivot levels (S1, S4) with volume confirmation and 4h EMA trend filter.
The Camarilla levels provide high-probability reversal/breakout points. Volume confirms institutional interest.
EMA filter ensures alignment with medium-term trend. Designed for fewer, high-quality trades to beat fee drag.
Targets 20-30 trades/year per symbol.
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
    
    # Calculate Camarilla pivot levels (S1, S4) from previous day
    # Formula: Pivot = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # S4 = C - (H - L) * 1.1 / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    s1 = close_1d - (range_hl * 1.1 / 12)
    s4 = close_1d - (range_hl * 1.1 / 2)
    
    # Get 4h data for EMA trend filter
    ema34 = np.full(n, np.nan)
    if n >= 34:
        ema34[33] = np.mean(close[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, n):
            ema34[i] = close[i] * alpha + ema34[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    # Align 1d Camarilla levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(s1_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema34[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above S1 with volume spike and 4h uptrend
            if (close[i] > s1_aligned[i] and vol_spike[i] and 
                close[i] > ema34[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S4 with volume spike and 4h downtrend
            elif (close[i] < s4_aligned[i] and vol_spike[i] and 
                  close[i] < ema34[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below S4 or 4h trend turns down
            if (close[i] < s4_aligned[i] or close[i] < ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above S1 or 4h trend turns up
            if (close[i] > s1_aligned[i] or close[i] > ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Camarilla_Pivot_S1S4_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0