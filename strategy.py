#!/usr/bin/env python3
"""
1h_PivotRange_MeanReversion_v1
Hypothesis: In 1h timeframe, price tends to revert to the mean within the previous day's trading range (high-low). 
Buy near daily low, sell near daily high, with volume confirmation and session filter (08-20 UTC).
Uses daily pivot range for mean reversion levels, works in both ranging and trending markets by fading extremes.
Target: 20-30 trades/year to avoid fee drag.
"""

name = "1h_PivotRange_MeanReversion_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for pivot range
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily high-low range and midpoint
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Pivot range boundaries (previous day's range)
    pivot_high = daily_high  # previous day's high
    pivot_low = daily_low    # previous day's low
    pivot_mid = (daily_high + daily_low) / 2.0  # midpoint of range
    
    # Align daily pivot levels to 1h timeframe
    pivot_high_aligned = align_htf_to_ltf(prices, df_1d, pivot_high)
    pivot_low_aligned = align_htf_to_ltf(prices, df_1d, pivot_low)
    pivot_mid_aligned = align_htf_to_ltf(prices, df_1d, pivot_mid)
    
    # Get 1h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 24-period EMA (to avoid low-volume noise)
    vol_ema24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_filter = volume > vol_ema24 * 1.5
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    session_mask = (prices.index.hour >= 8) & (prices.index.hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need volume EMA (24)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_high_aligned[i]) or 
            np.isnan(pivot_low_aligned[i]) or
            np.isnan(pivot_mid_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_mask[i]:
            if position != 0:
                signals[i] = 0.0  # close position outside session
                position = 0
            continue
        
        if position == 0:
            # Long: near daily low with volume confirmation
            if low[i] <= pivot_low_aligned[i] * 1.002 and volume_filter[i]:  # within 0.2% of low
                signals[i] = 0.20
                position = 1
            # Short: near daily high with volume confirmation
            elif high[i] >= pivot_high_aligned[i] * 0.998 and volume_filter[i]:  # within 0.2% of high
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: reach midpoint or reverse to high
            if close[i] >= pivot_mid_aligned[i] or high[i] >= pivot_high_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: reach midpoint or reverse to low
            if close[i] <= pivot_mid_aligned[i] or low[i] <= pivot_low_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals