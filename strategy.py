#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with daily volume confirmation and ATR filter.
Long when price breaks above R1 with volume spike and ATR rising; short when breaks below S1 with volume spike and ATR rising.
Uses daily Camarilla levels for structure, volume for confirmation, ATR for trend strength and stop.
Works in bull/bear: ATR filter avoids ranging markets, volume confirms institutional interest.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 5:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R1, S1)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Camarilla: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    r1_daily = close_daily + (high_daily - low_daily) * 1.1 / 12
    s1_daily = close_daily - (high_daily - low_daily) * 1.1 / 12
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    
    # Calculate ATR(14) for trend filter
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first TR
    
    atr = np.zeros(n)
    if n >= 14:
        atr[13] = np.mean(tr[:14])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume ratio (current vs 20-period average)
    vol_ma = np.zeros(n)
    if n >= 20:
        vol_ma[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    vol_ratio = np.zeros(n)
    for i in range(n):
        if vol_ma[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure volume MA and ATR are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike and ATR rising
            if (close[i] > r1_aligned[i] and 
                vol_ratio[i] > 1.5 and 
                atr[i] > atr[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and ATR rising
            elif (close[i] < s1_aligned[i] and 
                  vol_ratio[i] > 1.5 and 
                  atr[i] > atr[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S1 or ATR falling
            if close[i] < s1_aligned[i] or atr[i] < atr[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R1 or ATR falling
            if close[i] > r1_aligned[i] or atr[i] < atr[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals