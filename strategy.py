#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_S1_S4_Breakout
4h strategy using Camarilla pivot levels from 1d: S1 and S4 levels.
- Long: Close breaks above S4 (resistance) + volume > 1.5x daily avg
- Short: Close breaks below S1 (support) + volume > 1.5x daily avg
- Exit: Opposite breakout or reversal to opposite Camarilla level (S1 for longs, S4 for shorts)
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Daily OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: S1, S2, S3, S4, R1, R2, R3, R4
    # Range = daily high - daily low
    # S1 = close - (range * 1.05 / 6)
    # S2 = close - (range * 1.10 / 6)
    # S3 = close - (range * 1.15 / 6)
    # S4 = close - (range * 1.20 / 6)
    # R4 = close + (range * 1.20 / 6)
    # R3 = close + (range * 1.15 / 6)
    # R2 = close + (range * 1.10 / 6)
    # R1 = close + (range * 1.05 / 6)
    daily_range = high_1d - low_1d
    s1 = close_1d - (daily_range * 1.05 / 6)
    s2 = close_1d - (daily_range * 1.10 / 6)
    s3 = close_1d - (daily_range * 1.15 / 6)
    s4 = close_1d - (daily_range * 1.20 / 6)
    r4 = close_1d + (daily_range * 1.20 / 6)
    r3 = close_1d + (daily_range * 1.15 / 6)
    r2 = close_1d + (daily_range * 1.10 / 6)
    r1 = close_1d + (daily_range * 1.05 / 6)
    
    # Use S1 (support) and S4 (resistance) for breakouts
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(s1_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > s4_aligned[i]  # break above S4 resistance
        breakdown_down = close[i] < s1_aligned[i]  # break below S1 support
        
        if position == 0:
            # Long: volume + breakout above S4
            if vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: volume + breakdown below S1
            elif vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below S1 (opposite support) or reversal signal
            if breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above S4 (opposite resistance) or reversal signal
            if breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_S1_S4_Breakout"
timeframe = "4h"
leverage = 1.0