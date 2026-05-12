#!/usr/bin/env python3
name = "12h_WeeklyPivot_StructureBreak_VolumeFilter"
timeframe = "12h"
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
    
    # Weekly pivot points (calculated from previous week)
    df_1w = get_htf_data(prices, '1w')
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly pivot levels
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # Weekly trend: close above/below pivot
    weekly_trend_up = close_w > pivot_w
    weekly_trend_down = close_w < pivot_w
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Daily volume filter: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / vol_ma
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure volume MA has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or np.isnan(vol_ratio_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: only trade when volume is above average
        vol_filter = vol_ratio_aligned[i] > 1.5
        
        if position == 0:
            # Long: weekly trend up + price breaks above R1 + volume filter
            if (weekly_trend_up_aligned[i] and 
                close[i] > r1_w_aligned[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: weekly trend down + price breaks below S1 + volume filter
            elif (weekly_trend_down_aligned[i] and 
                  close[i] < s1_w_aligned[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly pivot OR weekly trend changes
            if close[i] < pivot_w_aligned[i] or not weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly pivot OR weekly trend changes
            if close[i] > pivot_w_aligned[i] or not weekly_trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals