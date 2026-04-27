#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot bias and volume confirmation.
# Uses weekly pivot levels (R1/S1) to filter breakout direction for higher probability trades.
# In bull markets, only take long breakouts above weekly R1; in bear markets, only short below weekly S1.
# Volume confirmation filters out false breakouts. Designed for 15-30 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot_w = np.full(len(df_1w), np.nan)
    r1_w = np.full(len(df_1w), np.nan)
    s1_w = np.full(len(df_1w), np.nan)
    
    for i in range(1, len(df_1w)):
        # Use previous week's data to calculate current week's pivot
        pp = (high_w[i-1] + low_w[i-1] + close_w[i-1]) / 3.0
        pivot_w[i] = pp
        r1_w[i] = 2 * pp - low_w[i-1]
        s1_w[i] = 2 * pp - high_w[i-1]
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # Donchian channel (20-period) on 6h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_confirmed = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or
            np.isnan(s1_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND above weekly R1 with volume
            if (close[i] > highest_high[i] and 
                close[i] > r1_w_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND below weekly S1 with volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < s1_w_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or weekly S1
            if (close[i] < lowest_low[i] or 
                close[i] < s1_w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or weekly R1
            if (close[i] > highest_high[i] or 
                close[i] > r1_w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_R1S1_Volume_v1"
timeframe = "6h"
leverage = 1.0