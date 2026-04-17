#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Regime_v1
Camarilla pivot breakout with volume confirmation and Choppiness regime filter.
Designed to work in both bull and bear markets by avoiding range-bound conditions.
Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # === 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots using previous day's OHLC
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1 = close_1d + (range_1d * 1.1 / 12)
    S1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align to 4h timeframe (previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # === 4h Volume Confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === 1d Choppiness Index (14-period) ===
    atr_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = tr
    
    # Smooth ATR
    atr_smoothed = np.full_like(close_1d, np.nan)
    for i in range(len(atr_1d)):
        if i >= 14:
            if i == 14:
                atr_smoothed[i] = np.mean(atr_1d[0:15])
            else:
                atr_smoothed[i] = (atr_smoothed[i-1] * 13 + atr_1d[i]) / 14
    
    # Calculate Chop
    sum_atr = np.zeros_like(close_1d)
    max_h = np.zeros_like(close_1d)
    min_l = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        if i >= 14:
            sum_atr[i] = np.sum(atr_smoothed[i-13:i+1])
            max_h[i] = np.max(high_1d[i-13:i+1])
            min_l[i] = np.min(low_1d[i-13:i+1])
            range_14 = max_h[i] - min_l[i]
            chop[i] = 100 * np.log10(sum_atr[i] / range_14) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_confirm[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 AND volume confirmation AND trending market (Chop < 38.2)
            if (close[i] > R1_aligned[i] and 
                vol_confirm[i] and 
                chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 AND volume confirmation AND trending market (Chop < 38.2)
            elif (close[i] < S1_aligned[i] and 
                  vol_confirm[i] and 
                  chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below pivot OR Chop > 61.8 (ranging market)
            if (close[i] < pivot_aligned[i] or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above pivot OR Chop > 61.8 (ranging market)
            if (close[i] > pivot_aligned[i] or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0