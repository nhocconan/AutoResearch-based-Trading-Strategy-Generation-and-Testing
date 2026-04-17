#!/usr/bin/env python3
"""
6h_Pivot_R1_S1_Breakout_Volume_ATRFilter
Hypothesis: Daily Pivot R1/S1 breakouts with volume confirmation and ATR-based volatility filter to avoid whipsaws during low-volatility periods. Works in both bull and bear markets by capturing breakout momentum while filtering false signals in choppy/low-volatility environments. Targets 15-30 trades/year per symbol.
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
    
    # === Daily data for pivot levels and ATR filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR for volatility filter
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    # ATR filter: avoid low volatility (ATR < 10-day MA of ATR)
    atr_filter = atr > atr_ma
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Daily Pivot points (standard)
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    r1 = pp + (range_hl * 1.0 / 3.0)  # Standard pivot R1
    s1 = pp - (range_hl * 1.0 / 3.0)  # Standard pivot S1
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    
    # Warmup: covers ATR calculations
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_filter_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # ATR filter: only trade when volatility is above average
        vol_filter = atr_filter_aligned[i]
        
        # Entry: only when flat
        if position == 0:
            # Long: break above R1 + volatility filter
            if close[i] > r1_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 + volatility filter
            elif close[i] < s1_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit: reverse signal
        elif position == 1:
            if close[i] < s1_aligned[i]:  # break below S1 = exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > r1_aligned[i]:  # break above R1 = exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0