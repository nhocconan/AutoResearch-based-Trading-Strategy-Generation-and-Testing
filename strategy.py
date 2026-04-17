#/usr/bin/env python3
"""
6h_WeeklyPivot_R1S1_Reverse_Volume_Filter_v1
Hypothesis: In BTC/ETH, price often reverses at weekly pivot S1/R1 levels during ranging markets (2022-2024). 
Weak bounces fail; strong reversals show volume confirmation. Uses weekly pivot S1/R1 as dynamic support/resistance.
In trending markets, avoids counter-trend trades by requiring price to be within weekly pivot R1-S1 range.
Timeframe: 6h balances noise and signal quality. Target: 60-120 trades over 4 years (15-30/year).
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
    
    # === Weekly Pivot Calculation (using prior week's OHLC) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P - H, R1 = 2P - L
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    s1 = 2 * pivot - weekly_high
    r1 = 2 * pivot - weekly_low
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1]) if i > 0 else volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.8  # volume spike: 1.8x average
    
    signals = np.zeros(n)
    
    # Warmup: need at least 1 weekly bar + 20 for volume MA
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Only trade when price is between S1 and R1 (range-bound condition)
        in_range = (s1_aligned[i] <= close[i] <= r1_aligned[i])
        
        # Entry logic: only enter when flat
        if position == 0 and in_range:
            # Long: price at or below S1 with volume confirmation (bounce from support)
            if close[i] <= s1_aligned[i] * 1.002 and vol_confirm[i]:  # 0.2% buffer
                signals[i] = 0.25
                position = 1
                continue
            # Short: price at or above R1 with volume confirmation (rejection at resistance)
            elif close[i] >= r1_aligned[i] * 0.998 and vol_confirm[i]:  # 0.2% buffer
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price reaches pivot (take profit) OR closes below S1 (breakdown)
            if (close[i] >= pivot_aligned[i] * 0.998 or  # near pivot
                close[i] < s1_aligned[i] * 0.998):       # break below S1
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches pivot (take profit) OR closes above R1 (breakout)
            if (close[i] <= pivot_aligned[i] * 1.002 or  # near pivot
                close[i] > r1_aligned[i] * 1.002):       # break above R1
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_Reverse_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0