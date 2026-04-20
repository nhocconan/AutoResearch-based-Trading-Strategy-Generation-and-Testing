#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_Volume_Trend_Filter
Hypothesis: Trade 4h price breakouts above/below daily pivot resistance/support levels with volume confirmation and 1d trend filter.
Long when price breaks above daily R1 with volume spike and 1d uptrend; short when breaks below daily S1 with volume spike and 1d downtrend.
Uses daily pivot levels (calculated from prior daily bar) and volume > 1.5x 20-period average for confirmation.
Designed for 4h timeframe to capture medium-term moves while reducing noise.
Target: 75-200 total trades over 4 years (19-50/year) with position size 0.25.
Works in bull/bear: 1d trend filter avoids counter-trend trades, volume filter reduces false breakouts.
"""

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_Trend_Filter"
timeframe = "4h"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (using prior daily bar's high, low, close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculation: PP = (H + L + C) / 3
    # R1 = 2*PP - L, S1 = 2*PP - H
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pp_1d - low_1d
    s1_1d = 2 * pp_1d - high_1d
    
    # Align daily pivot levels to 4h timeframe (already delayed by one bar via align_htf_to_ltf)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate volume filter (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above daily R1 with volume filter AND 1d uptrend (close > PP)
            if close[i] > r1_1d_aligned[i] and volume_filter[i] and close[i] > pp_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S1 with volume filter AND 1d downtrend (close < PP)
            elif close[i] < s1_1d_aligned[i] and volume_filter[i] and close[i] < pp_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below daily pivot point OR below S1
            if close[i] < pp_1d_aligned[i] or close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above daily pivot point OR above R1
            if close[i] > pp_1d_aligned[i] or close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals