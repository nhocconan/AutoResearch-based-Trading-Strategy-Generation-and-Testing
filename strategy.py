#!/usr/bin/env python3
"""
12h_1d_Pivot_R1_S1_Breakout_Volume_ATRFilter_V1
Long when price breaks above R1 with volume > 1.5x 20-period avg, short when breaks below S1.
Trend filter: price above/below 1d EMA50. Exit when price crosses back to pivot point.
Uses 12h timeframe for entries, 1d for pivot levels and trend filter.
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
    
    # === 1d Pivot Points (Classic) ===
    df_1d = get_htf_data(prices, '1d')
    # Ensure we have enough data
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate pivot points from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume filter: 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1, volume confirmation, price above 1d EMA50
            if (close[i] > r1_aligned[i] and 
                vol_condition and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1, volume confirmation, price below 1d EMA50
            elif (close[i] < s1_aligned[i] and 
                  vol_condition and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses back below pivot point
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above pivot point
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0