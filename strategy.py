#!/usr/bin/env python3
# 4H_1D_KAMA_Trend_With_Chop_Filter
# Hypothesis: On 4h timeframe, use KAMA to detect trend direction on 1d timeframe, enter long when 4h price crosses above KAMA in uptrend, short when crosses below in downtrend.
# Filter trades using Choppiness Index (CHOP) on 1d: only trade when CHOP < 61.8 (trending market).
# Uses volume confirmation: current volume > 1.5x 20-period average.
# Aims for 20-40 trades/year per symbol by combining trend, volatility regime, and volume filters.

name = "4H_1D_KAMA_Trend_With_Chop_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend and regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d KAMA trend (ER=10, fast=2, slow=30)
    change = np.abs(close_1d - np.roll(close_1d, 10))
    change[0:10] = 0  # first 10 values have no 10-period change
    volatility = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1d Choppiness Index (CHOP) - using 14-period
    atr_1d = np.zeros_like(close_1d)
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_hh - min_ll) > 0, 
                    100 * np.log10(atr_1d.sum() / (max_hh - min_ll)) / np.log10(14), 
                    50)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align 1d indicators to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in trending markets (CHOP < 61.8)
        if chop_aligned[i] >= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price crosses above KAMA + volume confirmation
            if close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below KAMA + volume confirmation
            elif close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals