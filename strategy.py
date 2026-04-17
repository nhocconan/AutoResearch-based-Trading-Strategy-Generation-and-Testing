#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Spike_Chop_Filter_v1
Camarilla pivot levels from 1d with volume spike and chop regime filter.
Long at R1 breakout with volume and chop > 61.8 (range). Short at S1 breakdown.
Uses 1d Camarilla levels for structure, 4h for entry timing.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === 1d Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels: R1, R2, R3, R4 and S1, S2, S3, S4
    r1 = close_1d + range_hl * 1.1 / 12
    s1 = close_1d - range_hl * 1.1 / 12
    
    # Align to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h Chopiness Index (14-period) ===
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14 if not np.isnan(atr[i-1]) else tr[i]
    
    # Sum of true range over 14 periods
    sum_tr = np.zeros(n)
    for i in range(n):
        if i >= 13:
            sum_tr[i] = np.sum(tr[i-13:i+1])
        else:
            sum_tr[i] = np.sum(tr[0:i+1]) if i > 0 else tr[0]
    
    # Max and min close over 14 periods
    max_close = np.zeros(n)
    min_close = np.zeros(n)
    for i in range(n):
        if i >= 13:
            max_close[i] = np.max(high[i-13:i+1])
            min_close[i] = np.min(low[i-13:i+1])
        else:
            max_close[i] = np.max(high[0:i+1]) if i > 0 else high[0]
            min_close[i] = np.min(low[0:i+1]) if i > 0 else low[0]
    
    # Chopiness index
    chop = np.full(n, 50.0)
    for i in range(n):
        if sum_tr[i] > 0 and max_close[i] > min_close[i]:
            chop[i] = 100 * np.log10(sum_tr[i] / (max_close[i] - min_close[i])) / np.log10(14)
    
    # === 4h Volume spike (20-period average) ===
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume and chop > 61.8 (range)
            if (close[i] > r1_aligned[i] and 
                vol_confirm[i] and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume and chop > 61.8 (range)
            elif (close[i] < s1_aligned[i] and 
                  vol_confirm[i] and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below S1 or chop < 38.2 (trend)
            if (close[i] < s1_aligned[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above R1 or chop < 38.2 (trend)
            if (close[i] > r1_aligned[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Spike_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0