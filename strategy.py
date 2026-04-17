#!/usr/bin/env python3
"""
12h_1d_Pivot_R1_S1_Breakout_Volume_MR
Strategy: 12h Camarilla Pivot (R1/S1) breakout with volume confirmation and mean-reversion filter.
- Long when price breaks above R1 + volume > 1.8x 12-period avg + price < S1 (mean-reversion setup)
- Short when price breaks below S1 + volume > 1.8x 12-period avg + price > R1 (mean-reversion setup)
- Exit when price reaches opposite pivot level (S1 for long, R1 for short) or midpoint
- Position size: ±0.25
- Uses 12h timeframe as primary with 1d for Pivot levels
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
    
    # Get 1d data for Pivot levels (Camarilla)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align 1d Pivot levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation (12-period MA on 12h)
    volume_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(12, 12)  # volume MA12, 1d data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(volume_ma12[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 12-period average
        volume_filter = volume[i] > (1.8 * volume_ma12[i])
        
        # Breakout conditions
        breakout_r1 = close[i] > r1_1d_aligned[i]  # break above R1
        breakout_s1 = close[i] < s1_1d_aligned[i]  # break below S1
        
        # Mean-reversion condition: price near opposite level
        near_s1 = abs(close[i] - s1_1d_aligned[i]) < 0.005 * close[i]  # within 0.5% of S1
        near_r1 = abs(close[i] - r1_1d_aligned[i]) < 0.005 * close[i]  # within 0.5% of R1
        
        if position == 0:
            # Long: breakout above R1 + volume filter + price near S1 (mean-reversion setup)
            if breakout_r1 and volume_filter and near_s1:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + price near R1 (mean-reversion setup)
            elif breakout_s1 and volume_filter and near_r1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches S1 or breaks below R1
            if near_s1 or close[i] < r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches R1 or breaks above S1
            if near_r1 or close[i] > s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R1_S1_Breakout_Volume_MR"
timeframe = "12h"
leverage = 1.0