#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Regime_v1
Camarilla Pivot R1/S1 breakout on 4h with volume confirmation and Choppiness regime filter.
Enters long when price breaks above R1 with volume spike in low-chop regime (trending).
Enters short when price breaks below S1 with volume spike in low-chop regime.
Uses 1d Camarilla levels for stability and to reduce noise.
Target: 20-50 trades per year (80-200 total over 4 years).
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
    
    # === 1d OHLC for Camarilla calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR1) / (n * (max(high) - min(low)))) / log10(n)
    # We'll use 14-period CHOP on daily data
    atr_1d = np.zeros_like(close_1d)
    tr_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            tr_1d[i] = high_1d[i] - low_1d[i]
        else:
            tr_1d[i] = max(high_1d[i] - low_1d[i], 
                           abs(high_1d[i] - close_1d[i-1]), 
                           abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = tr_1d[i]
    
    # Smooth ATR with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_smoothed = np.zeros_like(close_1d)
    if len(close_1d) >= 14:
        atr_smoothed[13] = np.mean(atr_1d[:14])
        for i in range(14, len(close_1d)):
            atr_smoothed[i] = (atr_smoothed[i-1] * 13 + atr_1d[i]) / 14
    
    # Calculate CHOP(14)
    chop = np.full_like(close_1d, 50.0)  # default to neutral
    for i in range(13, len(close_1d)):
        sum_atr = np.sum(atr_smoothed[i-13:i+1])
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        if max_high > min_low and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10((max_high - min_low) * 14)
        else:
            chop[i] = 50.0
    
    # === 4h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === Align 1d indicators to 4h timeframe ===
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat and in low-chop regime (trending)
        if position == 0 and chop_aligned[i] < 38.2:  # trending regime
            # Long: price breaks above R1 with volume confirmation
            if close[i] > r1_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < s1_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below S1 (reversal signal) OR chop becomes high (range)
            if close[i] < s1_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (reversal signal) OR chop becomes high (range)
            if close[i] > r1_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0