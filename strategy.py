#!/usr/bin/env python3
# 4h_12h_Camarilla_R1S1_MeanReversion_VolumeFilter
# Hypothesis: Mean reversion at daily Camarilla R1/S1 levels on 4h timeframe with 12h volume confirmation.
# In ranging markets (common in 2025), price tends to revert from extreme levels (R1/S1) back toward the mean (pivot).
# Uses 12h volume spike to confirm rejection at extremes. Works in both bull and bear as it fades extremes rather than following trends.
# Target: 15-30 trades/year per symbol to minimize fee drag.

name = "4h_12h_Camarilla_R1S1_MeanReversion_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate 12h volume average for spike detection
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 24:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=24, min_periods=24).mean().values  # 24*4h = 4 days
    
    # Align 1d and 12h indicators to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.8 * 12h average volume
        volume_spike = volume[i] > 1.8 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Long: price rejects S1 with volume spike (mean reversion up)
            if close[i] < s1_1d_aligned[i] * 1.005 and close[i] > s1_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price rejects R1 with volume spike (mean reversion down)
            elif close[i] > r1_1d_aligned[i] * 0.995 and close[i] < r1_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price reaches pivot or breaks above R1
            if close[i] >= pivot_1d_aligned[i] or close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price reaches pivot or breaks below S1
            if close[i] <= pivot_1d_aligned[i] or close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals