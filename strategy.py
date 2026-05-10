#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Go long when price touches Camarilla S1 (support) on 12h and closes above it with volume > 1.5x average, and daily trend is up (close > EMA34). Go short when price touches R1 (resistance) and closes below it with volume > 1.5x average, and daily trend is down (close < EMA34). Exit when price reverts to the Camarilla pivot (midpoint). Uses Camarilla levels from daily timeframe for structure, volume confirmation for conviction, and daily EMA for trend filter. Designed for 12h timeframe to target 15-30 trades/year per symbol. Works in bull markets by buying dips in uptrend and in bear markets by selling rallies in downtrend, avoiding ranging markets via volume filter.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate average volume for confirmation (50-period SMA)
    vol_avg = np.full(n, np.nan)
    for i in range(50, n):
        vol_avg[i] = np.mean(volume[i-50:i])
    
    # Calculate Camarilla levels from daily timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    cp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    pivot_1d = cp_1d  # use as exit level
    
    # Align to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure volume average and EMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(vol_avg[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches S1 and closes above it with volume confirmation and daily uptrend
            if low[i] <= s1_1d_aligned[i] and close[i] > s1_1d_aligned[i] and volume[i] > 1.5 * vol_avg[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 and closes below it with volume confirmation and daily downtrend
            elif high[i] >= r1_1d_aligned[i] and close[i] < r1_1d_aligned[i] and volume[i] > 1.5 * vol_avg[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price reverts to daily pivot (mean reversion to mean)
            if close[i] > pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reverts to daily pivot
            if close[i] < pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals