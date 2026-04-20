#!/usr/bin/env python3
# 6h_1d_WeeklyPivot_TrendFollowing
# Hypothesis: Trade in direction of weekly pivot trend using 6h price action with volume confirmation.
# Uses weekly pivot points (calculated from weekly OHLC) to determine trend bias:
#   - Above weekly pivot: long bias
#   - Below weekly pivot: short bias
# Entry occurs on 6h breakouts of recent highs/lows with volume surge.
# Designed for 12-37 trades per year by requiring weekly trend alignment + volume confirmation.
# Works in bull/bear markets by following the weekly trend.

name = "6h_1d_WeeklyPivot_TrendFollowing"
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
    
    # Get weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's data
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly pivot point and support/resistance levels
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    
    # Weekly support and resistance levels
    s1_w = 2 * pivot_w - high_w
    r1_w = 2 * pivot_w - low_w
    s2_w = pivot_w - range_w
    r2_w = pivot_w + range_w
    
    # Align weekly levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    
    # 6h Donchian channels (20-period) for breakout signals
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(r1_w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long bias: price above weekly pivot
            if close[i] > pivot_w_aligned[i]:
                # Long entry: break above 6h Donchian high with volume surge
                if close[i] > donchian_high[i] and volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = 0.25
                    position = 1
            # Short bias: price below weekly pivot
            else:
                # Short entry: break below 6h Donchian low with volume surge
                if close[i] < donchian_low[i] and volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly support or reverses below pivot
            if close[i] < s1_w_aligned[i] or close[i] < pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly resistance or reverses above pivot
            if close[i] > r1_w_aligned[i] or close[i] > pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals