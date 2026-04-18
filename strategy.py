#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume
Camarilla pivot breakout strategy:
- Long when price breaks above R1 level + volume confirmation
- Short when price breaks below S1 level + volume confirmation
- Exit on opposite S1/R1 break (reverse position)
- Uses 1d Camarilla levels from prior day
- Designed for 15-25 trades/year per symbol
Works in bull (breakouts) and bear (breakdowns) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day."""
    range_val = high - low
    c = close
    r4 = c + range_val * 1.1 / 2
    r3 = c + range_val * 1.1 / 4
    r2 = c + range_val * 1.1 / 6
    r1 = c + range_val * 1.1 / 12
    s1 = c - range_val * 1.1 / 12
    s2 = c - range_val * 1.1 / 6
    s3 = c - range_val * 1.1 / 4
    s4 = c - range_val * 1.1 / 2
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 for each day
    r1_1d = np.zeros(len(high_1d))
    s1_1d = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        r1, s1 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate volume moving average (10-period)
    vol_ma = np.full(len(volume), np.nan)
    if len(volume) >= 10:
        for i in range(9, len(volume)):
            vol_ma[i] = np.mean(volume[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 15  # need sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 10-period average
        vol_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 level + volume
            if close[i] > r1_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 level + volume
            elif close[i] < s1_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 level (reverse to short)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 level (reverse to long)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0