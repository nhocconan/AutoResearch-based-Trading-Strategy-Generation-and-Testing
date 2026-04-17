#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1
Camarilla pivot breakout on 12h timeframe with volume confirmation and chop regime filter.
Uses daily Camarilla pivot levels (R1, S1) for entry, volume spike for confirmation,
and Choppiness Index to avoid ranging markets.
Breakouts in trending markets (CHOP < 61.8) have higher success rate.
Designed to work in both bull and bear markets by following breakouts with trend.
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
    
    # === Daily Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1 = close_1d + (range_1d * 1.1 / 12)
    S1 = close_1d - (range_1d * 1.1 / 12)
    R2 = close_1d + (range_1d * 1.1 / 6)
    S2 = close_1d - (range_1d * 1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # === 12h Choppiness Index (14-period) ===
    atr_14 = np.full_like(close, np.nan)
    for i in range(n):
        if i >= 14:
            tr = np.max([
                high[i] - low[i],
                np.abs(high[i] - close[i-1]),
                np.abs(low[i] - close[i-1])
            ])
            if i == 14:
                atr_14[i] = np.mean([
                    high[1:15] - low[1:15],
                    np.abs(high[1:15] - close[0:14]),
                    np.abs(low[1:15] - close[0:14])
                ])
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr) / 14
    
    chop = np.full_like(close, np.nan)
    for i in range(n):
        if i >= 14 and not np.isnan(atr_14[i]):
            sum_atr = 0
            for j in range(14):
                idx = i - j
                if idx >= 0 and not np.isnan(atr_14[idx]):
                    sum_atr += atr_14[idx]
            if sum_atr > 0:
                chop[i] = 100 * np.log10(sum_atr / (atr_14[i] * 14)) / np.log10(14)
            else:
                chop[i] = 50
        else:
            chop[i] = 50
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
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
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume AND trending market (CHOP < 61.8)
            if (close[i] > R1_aligned[i] and 
                vol_confirm[i] and 
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume AND trending market (CHOP < 61.8)
            elif (close[i] < S1_aligned[i] and 
                  vol_confirm[i] and 
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below S1 OR chop becomes too high (ranging)
            if (close[i] < S1_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 OR chop becomes too high (ranging)
            if (close[i] > R1_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0