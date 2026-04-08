#!/usr/bin/env python3
# 12h_camilla_pivot_volume_v3
# Hypothesis: Uses daily Camarilla pivot levels with volume confirmation on 12h timeframe.
# Long when price touches S3/S4 support with volume surge, short when price touches R3/R4 resistance with volume surge.
# Exits when price moves back toward median (Pivot) or volume drops.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
# Works in bull/bear via mean reversion at extreme pivot levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camilla_pivot_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Resistance levels
    R4 = close_1d + range_hl * 1.500
    R3 = close_1d + range_hl * 1.250
    R2 = close_1d + range_hl * 1.166
    R1 = close_1d + range_hl * 1.083
    
    # Support levels
    S1 = close_1d - range_hl * 1.083
    S2 = close_1d - range_hl * 1.166
    S3 = close_1d - range_hl * 1.250
    S4 = close_1d - range_hl * 1.500
    
    # Align levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    R4_12h = align_htf_to_ltf(prices, df_1d, R4)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume filter: 2.0x 24-period average (2 days of 12h data)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 2.0 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(R4_12h[i]) or np.isnan(R3_12h[i]) or 
            np.isnan(S4_12h[i]) or np.isnan(S3_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price moves back to pivot or volume drops
            if close[i] >= pivot_12h[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price moves back to pivot or volume drops
            if close[i] <= pivot_12h[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price at S3/S4 support with volume surge
            if vol_surge[i] and (
                (close[i] <= S3_12h[i] and low[i] <= S4_12h[i]) or  # Touched S4
                (close[i] <= S4_12h[i] and low[i] <= S3_12h[i])   # Touched S3
            ):
                position = 1
                signals[i] = 0.25
            # Short entry: Price at R3/R4 resistance with volume surge
            elif vol_surge[i] and (
                (close[i] >= R3_12h[i] and high[i] >= R4_12h[i]) or  # Touched R4
                (close[i] >= R4_12h[i] and high[i] >= R3_12h[i])   # Touched R3
            ):
                position = -1
                signals[i] = -0.25
    
    return signals