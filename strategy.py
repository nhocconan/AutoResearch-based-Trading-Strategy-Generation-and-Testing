#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Volume_Filter_v1
Breakout strategy using Camarilla pivot levels (R1, S1) from 1d timeframe for signal direction.
Enters long when price breaks above R1 with volume confirmation, short when breaks below S1.
Uses 1h for entry timing only. Includes session filter (08-20 UTC) to reduce noise.
Target: 60-150 total trades over 4 years (15-37/year).
"""

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
    
    # === Volume confirmation: 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d Camarilla pivot levels (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align to 1h timeframe (waits for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Pre-compute session hours (08-20 UTC) - avoid recomputing in loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_ma[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if close[i] > r1_aligned[i] and vol_confirmed:
                signals[i] = 0.20
                position = 1
                continue
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < s1_aligned[i] and vol_confirmed:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic: reverse signal or loss of momentum
        elif position == 1:
            # Exit long: price breaks below S1 (reverse signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above R1 (reverse signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0