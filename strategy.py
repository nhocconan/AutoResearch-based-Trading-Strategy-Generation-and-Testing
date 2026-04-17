#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_SwingReversal_v1
Camarilla pivot reversal strategy on 12h timeframe.
Uses daily Camarilla pivot levels (R1, S1) as key reversal zones.
Enters on rejection of these levels with volume confirmation.
Designed to work in both bull and bear markets by capturing mean reversion
at institutional pivot levels.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (using standard multipliers)
    R1 = pivot + (range_1d * 1.1 / 12)
    S1 = pivot - (range_1d * 1.1 / 12)
    R2 = pivot + (range_1d * 1.1 / 6)
    S2 = pivot - (range_1d * 1.1 / 6)
    
    # === 12h Volume Confirmation (24-period average) ===
    vol_ma_24 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 24:
            vol_ma_24[i] = np.mean(volume[i-23:i+1])
        elif i > 0:
            vol_ma_24[i] = np.mean(volume[max(0, i-12):i+1])
        else:
            vol_ma_24[i] = volume[0]
    
    vol_confirm = volume > vol_ma_24 * 1.8  # volume spike: 1.8x average
    
    # === Align 1d Camarilla levels to 12h timeframe ===
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price near S1 support with rejection (close > S1) and volume confirmation
            # Allow small penetration but require close back above S1
            if (close[i] > S1_aligned[i] and 
                low[i] <= S1_aligned[i] * 1.005 and  # touched or went slightly below
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price near R1 resistance with rejection (close < R1) and volume confirmation
            elif (close[i] < R1_aligned[i] and 
                  high[i] >= R1_aligned[i] * 0.995 and  # touched or went slightly above
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price reaches pivot or R1, or volume confirmation fails
            if (close[i] >= pivot_aligned[i] or 
                close[i] >= R1_aligned[i] or
                not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches pivot or S1, or volume confirmation fails
            if (close[i] <= pivot_aligned[i] or 
                close[i] <= S1_aligned[i] or
                not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_SwingReversal_v1"
timeframe = "12h"
leverage = 1.0