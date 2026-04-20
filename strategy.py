#!/usr/bin/env python3
"""
6h_1D_Pivot_R2S2_MomentumBreakout
Hypothesis: Trade 6-hour breakouts with daily pivot levels. Go long when price breaks above R2 with volume confirmation, short when breaks below S2. 
This targets momentum continuation after breaking key daily support/resistance levels. Works in both bull and bear markets by capturing breakouts in the direction of the daily pivot structure.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Uses daily pivot points (R2/S2) as key institutional levels, volume confirmation to filter false breakouts, and momentum to ensure follow-through.
"""

name = "6h_1D_Pivot_R2S2_MomentumBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: P = (H+L+C)/3, R2 = P + (H-L), S2 = P - (H-L)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    pivot_daily = (high_daily + low_daily + close_daily) / 3.0
    range_daily = high_daily - low_daily
    r2_daily = pivot_daily + range_daily
    s2_daily = pivot_daily - range_daily
    
    # Align daily pivot levels to 6h timeframe (wait for daily close)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2_daily)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2_daily)
    
    # Calculate volume filter (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Calculate momentum (price change over 3 periods)
    momentum = np.full_like(close, np.nan)
    for i in range(3, n):
        momentum[i] = (close[i] - close[i-3]) / close[i-3]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(momentum[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R2 with volume and momentum
            if close[i] > r2_aligned[i] and volume_filter[i] and momentum[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume and momentum
            elif close[i] < s2_aligned[i] and volume_filter[i] and momentum[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below R2 or momentum turns negative
            if close[i] < r2_aligned[i] or momentum[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above S2 or momentum turns positive
            if close[i] > s2_aligned[i] or momentum[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals