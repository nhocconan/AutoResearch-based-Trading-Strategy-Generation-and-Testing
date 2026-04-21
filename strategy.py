#!/usr/bin/env python3
"""
4h_1D_Camarilla_R1S1_Breakout_VolumeFilter
Hypothesis: Daily Camarilla pivot levels R1/S1 act as mean-reversion zones on the 4h chart. Price returning to these levels with volume confirmation indicates exhaustion and reversal. Designed for low trade frequency (target: 25-40/year) to minimize fee drag. Works in bull markets via mean reversion at extremes and in bear markets via fade of overextended moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Camarilla pivot points
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily Camarilla pivot levels
    # P = (H + L + C) / 3
    # Range = H - L
    # R1 = P + (Range * 0.382)
    # S1 = P - (Range * 0.382)
    P = (high_daily + low_daily + close_daily) / 3.0
    range_daily = high_daily - low_daily
    r1_daily = P + (range_daily * 0.382)
    s1_daily = P - (range_daily * 0.382)
    
    # Align daily Camarilla levels to 4h timeframe
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.8x 20-period average (20*4h = 10 days)
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.8 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_daily_aligned[i]) or np.isnan(s1_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = r1_daily_aligned[i]
        s1 = s1_daily_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Fade at R1/S1: mean reversion from extreme levels with volume
            # Long: price rejects S1 with volume confirmation (buying pressure)
            if price > s1 and price < (s1 + (r1 - s1) * 0.4) and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price rejects R1 with volume confirmation (selling pressure)
            elif price < r1 and price > (r1 - (r1 - s1) * 0.4) and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to midpoint or breaks above R1
            midpoint = s1 + (r1 - s1) * 0.5
            if price < midpoint or price > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint or breaks below S1
            midpoint = s1 + (r1 - s1) * 0.5
            if price > midpoint or price < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1D_Camarilla_R1S1_Breakout_VolumeFilter"
timeframe = "4h"
leverage = 1.0