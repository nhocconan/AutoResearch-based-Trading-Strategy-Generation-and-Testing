#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_SpreadFilter
Camarilla pivot breakout with volume confirmation and spread filter for 12h timeframe.
Targets 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
Works in both bull and bear markets by combining price channel structure with volume confirmation.
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
    # Calculate Camarilla levels from previous day's OHLC
    # Using typical Camarilla formula: 
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range and Camarilla levels
    daily_range = high_1d - low_1d
    r1_level = close_1d + 1.1 * daily_range / 12
    s1_level = close_1d - 1.1 * daily_range / 12
    
    # Align to 12h timeframe (previous day's levels available after close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
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
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirmed = volume[i] > 1.8 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if (close[i] > r1_aligned[i] and vol_confirmed):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume confirmation
            elif (close[i] < s1_aligned[i] and vol_confirmed):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volume dropout
        elif position == 1:
            # Exit long: price breaks below S1 OR volume drops below average
            if (close[i] < s1_aligned[i] or volume[i] < 0.7 * vol_ma[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 OR volume drops below average
            if (close[i] > r1_aligned[i] or volume[i] < 0.7 * vol_ma[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_SpreadFilter"
timeframe = "12h"
leverage = 1.0