#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_Volume
Breakout above weekly pivot R1 or below S1 with volume confirmation.
Weekly pivot provides institutional levels; volume confirms institutional interest.
Designed to work in both bull and breakout phases of bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === Weekly Pivot Points (using Monday's OHLC) ===
    # Get weekly data
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate pivot points from previous week's OHLC
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # === Volume Spike Detection ===
    # Volume > 1.5x 20-period average indicates institutional interest
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume spike
            if (close[i] > r1_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume spike
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volatility exhaustion
        elif position == 1:
            # Exit long: price breaks below pivot OR opposite signal
            if (close[i] < pivot_aligned[i] or 
                (close[i] < s1_aligned[i] and volume_spike[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above pivot OR opposite signal
            if (close[i] > pivot_aligned[i] or 
                (close[i] > r1_aligned[i] and volume_spike[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0