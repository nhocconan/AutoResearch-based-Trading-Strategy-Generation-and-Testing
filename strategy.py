#!/usr/bin/env python3
"""
12h_KAMA_Trend_Reverse_Entry
Hypothesis: In ranging or choppy markets (common in 2025-2026), price often reverts to the KAMA trend after short-term extremes. 
We enter long when price crosses below KAMA (oversold in downtrend) and short when price crosses above KAMA (overbought in uptrend), 
using 1-day trend filter to avoid counter-trend trades. Volume spike confirms momentum. 
Designed for low trade frequency (<30/year) to minimize fee decay in sideways markets.
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
    
    # Calculate 1-day KAMA trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio (ER) over 10 periods
    change_10 = np.abs(np.subtract(close_1d[10:], close_1d[:-10]))
    abs_change = np.sum(np.abs(np.diff(close_1d, axis=0))[:len(close_1d)-10:], axis=0) if len(close_1d) > 10 else np.array([])
    er = np.full(len(close_1d), np.nan)
    if len(change_10) > 0:
        er[10:] = change_10 / np.maximum(abs_change, 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.full(len(close_1d), np.nan)
    if len(close_1d) > 10:
        kama[10] = close_1d[10]  # seed
        for i in range(11, len(close_1d)):
            if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align 1-day KAMA to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1-day trend filter: price > KAMA = uptrend, price < KAMA = downtrend
    # We'll use this to filter entries only
    
    # Volume spike: current volume > 2.0 x 24-period average (more selective)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 10)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses below KAMA (oversold) in 1-day uptrend with volume spike
            if (close[i] <= kama_1d_aligned[i] and close[i-1] > kama_1d_aligned[i-1] and 
                close[i] > kama_1d_aligned[i] and vol_spike[i]):  # Wait for confirmation bar
                signals[i] = 0.25
                position = 1
            # Short: price crosses above KAMA (overbought) in 1-day downtrend with volume spike
            elif (close[i] >= kama_1d_aligned[i] and close[i-1] < kama_1d_aligned[i-1] and 
                  close[i] < kama_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back above KAMA (mean reversion complete) or trend fails
            if (close[i] >= kama_1d_aligned[i] and close[i-1] < kama_1d_aligned[i-1]) or close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back below KAMA or trend fails
            if (close[i] <= kama_1d_aligned[i] and close[i-1] > kama_1d_aligned[i-1]) or close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_Reverse_Entry"
timeframe = "12h"
leverage = 1.0