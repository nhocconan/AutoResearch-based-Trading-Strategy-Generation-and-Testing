#!/usr/bin/env python3
"""
6h_WeeklyPivot_RangeReversion
Hypothesis: Use weekly pivot points from 1w data to identify key support/resistance levels.
In range-bound markets (identified by 1d Choppiness Index > 61.8), mean revert from S1/R1 and S2/R2 levels.
In trending markets (Choppiness Index < 38.2), breakout trades from S2/R2 levels.
This adapts to both bull/bear regimes via the 1d chop filter and uses weekly structure for level significance.
Target: 15-30 trades/year on 6b timeframe.
"""

name = "6h_WeeklyPivot_RangeReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    r2 = pivot + (high_prev - low_prev)
    s2 = pivot - (high_prev - low_prev)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate 1d Choppiness Index for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    # ATR(14)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(tr_sum / (atr_14 * 14)) / log10(14)
    chop_raw = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    
    # Align chop to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Get 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot (needs 1w bar), chop (14 periods)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop = chop_aligned[i]
        
        if position == 0:
            # Range market: chop > 61.8 -> mean revert from S1/R1
            if chop > 61.8:
                # Long from S1 with rejection (low touches S1 but close above)
                if low[i] <= s1_aligned[i] and close[i] > s1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short from R1 with rejection (high touches R1 but close below)
                elif high[i] >= r1_aligned[i] and close[i] < r1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            # Trending market: chop < 38.2 -> breakout from S2/R2
            elif chop < 38.2:
                # Long breakout above R2
                if high[i] > r2_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below S2
                elif low[i] < s2_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit conditions
            if chop > 61.8:
                # In range: exit at pivot or R1
                if close[i] >= pivot_aligned[i] or close[i] >= r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In trend: trail with S2 or exit if chop turns to range
                if chop > 61.8 or low[i] < s2_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if chop > 61.8:
                # In range: exit at pivot or S1
                if close[i] <= pivot_aligned[i] or close[i] <= s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In trend: trail with R2 or exit if chop turns to range
                if chop > 61.8 or high[i] > r2_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals