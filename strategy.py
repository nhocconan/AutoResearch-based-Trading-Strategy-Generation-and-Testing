#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_Volume_TrendFilter
# Hypothesis: On 6h chart, break above weekly pivot R1 with volume spike and trend alignment = long.
# Break below weekly pivot S1 with volume spike and trend alignment = short.
# Uses weekly pivot levels from 1w data, volume confirmation > 1.5x 20-period average, and 6h EMA50 trend filter.
# Designed to work in both bull and bear markets by following the trend via EMA50.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WeeklyPivot_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1w data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 6h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50 and volume average
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema50[i]) or 
            np.isnan(vol_avg[i]) or vol_avg[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long: price breaks above R1, volume spike, and uptrend (price > EMA50)
            if (close[i] > r1_1w_aligned[i] and 
                volume_spike and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, volume spike, and downtrend (price < EMA50)
            elif (close[i] < s1_1w_aligned[i] and 
                  volume_spike and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below pivot or trend reverses
            if close[i] < pivot_1w_aligned[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above pivot or trend reverses
            if close[i] > pivot_1w_aligned[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals