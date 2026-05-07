#!/usr/bin/env python3
name = "6h_WeeklyPivot_RangeBreakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Pivot Point = (High + Low + Close) / 3
    pivot = (high_w + low_w + close_w) / 3
    # Support 1 = (2 * Pivot) - High
    s1 = (2 * pivot) - high_w
    # Resistance 1 = (2 * Pivot) - Low
    r1 = (2 * pivot) - low_w
    # Support 2 = Pivot - (High - Low)
    s2 = pivot - (high_w - low_w)
    # Resistance 2 = Pivot + (High - Low)
    r2 = pivot + (high_w - low_w)
    
    # Align weekly pivots to 6h timeframe (no extra delay - pivots are fixed for the week)
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    
    # Get daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if we're in a ranging market (between S1 and R1)
        in_range = (s1_aligned[i] <= close[i] <= r1_aligned[i])
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume, in uptrend
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume, in downtrend
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 (range reversion) or breaks above R2 (take profit)
            if (close[i] < s1_aligned[i] or 
                close[i] > r2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 (range reversion) or breaks below S2 (take profit)
            if (close[i] > r1_aligned[i] or 
                close[i] < s2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals