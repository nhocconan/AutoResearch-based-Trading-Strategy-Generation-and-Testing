#!/usr/bin/env python3
"""
6h_Camarilla_Pivot_Breakout_VolumeFilter
Camarilla pivot breakout with volume confirmation on 6h timeframe.
Uses 12h Camarilla pivot levels (R1-S1, R4-S4) for breakout signals.
Trades breakouts of R4/S4 with volume confirmation, mean reverts at R3/S3.
Designed for 6h timeframe to target 50-150 total trades over 4 years.
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
    
    # === 12h Camarilla pivot levels ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    R4 = close_12h + range_12h * 1.500
    R3 = close_12h + range_12h * 1.250
    R2 = close_12h + range_12h * 1.166
    R1 = close_12h + range_12h * 1.083
    
    S1 = close_12h - range_12h * 1.083
    S2 = close_12h - range_12h * 1.166
    S3 = close_12h - range_12h * 1.250
    S4 = close_12h - range_12h * 1.500
    
    # === 6h Volume confirmation (24-period average) ===
    vol_ma_24 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 24:
            vol_ma_24[i] = np.mean(volume[i-23:i+1])
        elif i > 0:
            vol_ma_24[i] = np.mean(volume[max(0, i-12):i+1])
        else:
            vol_ma_24[i] = volume[0]
    
    vol_confirm = volume > vol_ma_24 * 1.3  # volume spike: 1.3x average
    
    # === Align 12h Camarilla levels to 6h timeframe ===
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    R4_aligned = align_htf_to_ltf(prices, df_12h, R4)
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    S4_aligned = align_htf_to_ltf(prices, df_12h, S4)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long breakout: price breaks above R4 with volume
            if (close[i] > R4_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short breakdown: price breaks below S4 with volume
            elif (close[i] < S4_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
            # Mean reversion long: price touches S3 with rejection
            elif (low[i] <= S3_aligned[i] and 
                  close[i] > S3_aligned[i] * 1.002 and  # closed above S3
                  vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Mean reversion short: price touches R3 with rejection
            elif (high[i] >= R3_aligned[i] and 
                  close[i] < R3_aligned[i] * 0.998 and  # closed below R3
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below R3 or S4 breakdown
            if (close[i] < R3_aligned[i] or 
                close[i] < S4_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above S3 or R4 breakout
            if (close[i] > S3_aligned[i] or 
                close[i] > R4_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_Pivot_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0