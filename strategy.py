#!/usr/bin/env python3
"""
6h_Pivot_R1S1_Breakout_VolumeConfirmation_V1
Hypothesis: Daily pivot points (R1/S1) act as strong intraday support/resistance on 6h timeframe.
Breakouts above R1 or below S1 with volume confirmation (volume > 1.5x 20-period average) indicate
institutional interest and trend continuation. Works in bull/bear by only taking breakouts in
direction of price relative to central pivot (above PP = long bias, below PP = short bias).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate daily pivot points: P, R1, S1, R2, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1d data once for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate pivot points on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d, r1_1d, s1_1d, r2_1d, s2_1d = calculate_pivot_points(high_1d, low_1d, close_1d)
    
    # Align pivot levels to 6h timeframe (only use completed daily bars)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 6h data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any pivot level is NaN
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_average = vol_ma[i]
        
        # Skip if volume data not ready
        if np.isnan(vol_average):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol > (1.5 * vol_average)
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation AND price above daily pivot (bullish bias)
            if price > r1_aligned[i] and volume_confirmed and price > pivot_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation AND price below daily pivot (bearish bias)
            elif price < s1_aligned[i] and volume_confirmed and price < pivot_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or loses volume confirmation
            if price < r1_aligned[i] or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or loses volume confirmation
            if price > s1_aligned[i] or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1S1_Breakout_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0