#!/usr/bin/env python3
name = "1d_Weekly_Pivot_Range_Reversion"
timeframe = "1d"
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
    
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly pivot points (using previous week)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    
    # Weekly ATR for volatility filter
    tr1 = np.abs(df_1w['high'].values - df_1w['low'].values)
    tr2 = np.abs(df_1w['high'].values - np.roll(df_1w['close'].values, 1))
    tr3 = np.abs(df_1w['low'].values - np.roll(df_1w['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = np.zeros(len(tr))
    for i in range(len(tr)):
        if i < 14:
            atr_1w[i] = np.mean(tr[:i+1]) if i > 0 else 0
        else:
            atr_1w[i] = np.mean(tr[i-13:i+1])
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Align pivot levels to daily
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Daily range position (where price is within weekly range)
    range_width = r2_aligned - s2_aligned
    range_position = np.zeros(n)
    for i in range(n):
        if range_width[i] > 0:
            range_position[i] = (close[i] - s2_aligned[i]) / range_width[i]
        else:
            range_position[i] = 0.5
    
    # Volume filter: current volume > 1.3x 20-day average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Mean reversion at extremes with volume confirmation
            # Long near support in weekly range
            if (range_position[i] < 0.2 and 
                close[i] > s1_aligned[i] and 
                volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short near resistance in weekly range
            elif (range_position[i] > 0.8 and 
                  close[i] < r1_aligned[i] and 
                  volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: return to middle of range or break above resistance
            if (range_position[i] > 0.6 or close[i] > r2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: return to middle of range or break below support
            if (range_position[i] < 0.4 or close[i] < s2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals