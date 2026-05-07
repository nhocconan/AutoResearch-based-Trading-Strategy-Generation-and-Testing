#!/usr/bin/env python3
"""
6H_WeeklyPivot_DailyTrend_VolumeBreakout
Hypothesis: 6h price breaks above/below weekly pivot levels with daily EMA50 trend confirmation and volume spike.
Weekly pivots provide strong support/resistance that hold across market regimes. Daily EMA50 ensures
alignment with intermediate trend, reducing counter-trend trades. Volume confirmation validates breakout
strength. Targets 15-35 trades/year on 6h timeframe to minimize fee drag.
"""
name = "6H_WeeklyPivot_DailyTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    # Weekly resistance/support levels (standard pivot)
    r1_w = pivot_w * 2 - low_w      # R1 = 2*P - L
    s1_w = pivot_w * 2 - high_w     # S1 = 2*P - H
    r2_w = pivot_w + range_w        # R2 = P + (H-L)
    s2_w = pivot_w - range_w        # S2 = P - (H-L)
    
    # Align weekly levels to 6h timeframe (using prior week's values)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Get daily data for EMA50 trend and volume average
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend direction
    close_d = df_d['close'].values
    ema_50 = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_d, ema_50)
    
    # Daily average volume for volume filter
    vol_d = df_d['volume'].values
    vol_avg_d = pd.Series(vol_d).rolling(window=20, min_periods=20).mean().values
    vol_avg_d_aligned = align_htf_to_ltf(prices, df_d, vol_avg_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(r2_w_aligned[i]) or np.isnan(s2_w_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 72 bars between trades (18 days on 6h TF) to reduce frequency
            if bars_since_exit < 72:
                continue
                
            # Long: price breaks above weekly R2 with daily EMA50 uptrend and volume spike
            if (close[i] > r2_w_aligned[i] and close[i-1] <= r2_w_aligned[i-1] and 
                close[i] > ema_50_aligned[i] and volume[i] > vol_avg_d_aligned[i] * 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below weekly S2 with daily EMA50 downtrend and volume spike
            elif (close[i] < s2_w_aligned[i] and close[i-1] >= s2_w_aligned[i-1] and 
                  close[i] < ema_50_aligned[i] and volume[i] > vol_avg_d_aligned[i] * 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to weekly pivot level (mean reversion to pivot)
            if position == 1 and close[i] <= pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] >= pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals