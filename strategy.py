#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter
Hypothesis: Use 4h Camarilla pivot R1/S1 breakout for trend direction with 1d volume confirmation and 1h entry timing. 
Camarilla pivots provide precise support/resistance levels that work in both trending and ranging markets. 
Volume filter ensures moves have conviction. Limited to 1-2 trades per week to control fee drag.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_ = high - low
    r1 = close + range_ * 1.1 / 12
    s1 = close - range_ * 1.1 / 12
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot levels (trend direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels
    camarilla_pivot_4h = np.zeros(len(df_4h))
    camarilla_r1_4h = np.zeros(len(df_4h))
    camarilla_s1_4h = np.zeros(len(df_4h))
    
    for i in range(len(df_4h)):
        pivot, r1, s1 = calculate_camarilla(high_4h[i], low_4h[i], close_4h[i])
        camarilla_pivot_4h[i] = pivot
        camarilla_r1_4h[i] = r1
        camarilla_s1_4h[i] = s1
    
    # Align 4h Camarilla levels to 1h timeframe
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate 20-period average volume on daily timeframe
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    # Current day's volume > 1.5x 20-day average
    volume_filter_1d = volume_1d > (vol_ma_1d * 1.5)
    # Align volume filter to 1h timeframe
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_4h_aligned[i]) or 
            np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or 
            np.isnan(volume_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if close[i] > r1_4h_aligned[i] and volume_filter_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < s1_4h_aligned[i] and volume_filter_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot level
            if close[i] <= pivot_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price returns to pivot level
            if close[i] >= pivot_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals