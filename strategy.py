#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_v1
# Hypothesis: Combines weekly pivot points (from 1w) with 6h price action to capture breakouts.
# In both bull and bear markets, price often respects weekly support/resistance levels.
# Long when price breaks above weekly R1 with volume confirmation; short when breaks below weekly S1.
# Uses 6 Donchian channel to filter for momentum and avoid false breakouts.
# Target: 15-25 trades/year (60-100 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1. Weekly pivot points (using weekly OHLC)
    df_1w = get_htf_data(prices, '1w')  # Load weekly data ONCE
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot: P = (H + L + C) / 3
    # Resistance: R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly levels to 6h timeframe (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # 2. 6-period Donchian channel on 6h for momentum filter
    donchian_period = 6
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= donchian_period - 1:
            highest_high[i] = np.max(high[i-donchian_period+1:i+1])
            lowest_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    # 3. Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.3
        
        if position == 1:  # Long position
            # Exit: price closes below weekly pivot OR Donchian break down
            if close[i] < weekly_pivot_aligned[i] or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly pivot OR Donchian break up
            if close[i] > weekly_pivot_aligned[i] or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly R1 with volume and Donchian confirmation
            if (close[i] > weekly_r1_aligned[i] and 
                close[i] > highest_high[i] and 
                vol_ok):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly S1 with volume and Donchian confirmation
            elif (close[i] < weekly_s1_aligned[i] and 
                  close[i] < lowest_low[i] and 
                  vol_ok):
                position = -1
                signals[i] = -0.25
    
    return signals