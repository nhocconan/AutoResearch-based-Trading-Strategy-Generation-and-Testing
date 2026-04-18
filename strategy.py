#!/usr/bin/env python3
"""
12h_KAMA_Trend_Reversal_with_1dKAMA_Confirmation
Hypothesis: 12h KAMA crossover signals trend changes, confirmed by 1d KAMA direction.
Only trade when both timeframes agree to avoid whipsaw in choppy markets.
Designed for low trade frequency (<40/year) to minimize fee drag while capturing major trend reversals.
Works in both bull and bear markets by following the confirmed trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate 12h KAMA (primary signal)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    vol = np.concatenate([np.full(10, np.nan), vol])
    er = np.where(vol != 0, change / vol, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d KAMA for confirmation (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d KAMA
    change_1d = np.abs(np.diff(close_1d, n=5))  # 5-period change for daily
    vol_1d = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 5-period volatility
    change_1d = np.concatenate([np.full(5, np.nan), change_1d])
    vol_1d = np.concatenate([np.full(5, np.nan), vol_1d])
    er_1d = np.where(vol_1d != 0, change_1d / vol_1d, 0)
    sc_1d = (er_1d * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    kama_1d = np.full(len(close_1d), np.nan)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc_1d[i]):
            kama_1d[i] = kama_1d[i-1]
        else:
            kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (close_1d[i] - kama_1d[i-1])
    
    # Align 1d KAMA to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Generate signals
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need enough warmup for KAMA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(kama[i-10]) or 
            np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1d_aligned[i-1])):
            signals[i] = 0.0
            continue
        
        # 12h KAMA crossover signals
        kama_cross_up = kama[i] > kama[i-10] and kama[i-1] <= kama[i-10]
        kama_cross_down = kama[i] < kama[i-10] and kama[i-1] >= kama[i-10]
        
        # 1d KAMA direction (trend filter)
        kama_1d_up = kama_1d_aligned[i] > kama_1d_aligned[i-1]
        kama_1d_down = kama_1d_aligned[i] < kama_1d_aligned[i-1]
        
        if position == 0:
            # Long: 12h KAMA crosses up + 1d KAMA rising
            if kama_cross_up and kama_1d_up:
                signals[i] = 0.25
                position = 1
            # Short: 12h KAMA crosses down + 1d KAMA falling
            elif kama_cross_down and kama_1d_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: 12h KAMA crosses down OR 1d KAMA turns down
            if kama_cross_down or not kama_1d_up:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: 12h KAMA crosses up OR 1d KAMA turns up
            if kama_cross_up or kama_1d_up:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_Reversal_with_1dKAMA_Confirmation"
timeframe = "12h"
leverage = 1.0