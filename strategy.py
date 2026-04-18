#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_RangeFilter_v1
Hypothesis: Trade breakouts of weekly R1/S1 levels with 1d range filter to avoid false breakouts.
Uses weekly pivot for structural levels and 1d ATR-based range filter to distinguish trending vs ranging markets.
Target: 20-30 trades/year per symbol. Works in bull/bear via range filter that avoids chop.
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
    
    # Get weekly data for pivot levels
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot and R1/S1
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    r1_w = pivot_w + range_w * 1.1 / 12
    s1_w = pivot_w - range_w * 1.1 / 12
    
    # Align weekly levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_weekly, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_weekly, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_weekly, s1_w)
    
    # Get daily data for range filter (ATR-based)
    df_daily = get_htf_data(prices, '1d')
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # Calculate 14-day ATR for range filter
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = np.full(n, np.nan)
    for i in range(14, len(tr)):
        atr_14d[i] = np.mean(tr[i-13:i+1])
    
    # Align ATR to 6h timeframe
    atr_14d_aligned = align_htf_to_ltf(prices, df_daily, atr_14d)
    
    # Calculate 6-day average true range for normalization
    tr_6h1 = high[1:] - low[1:]
    tr_6h2 = np.abs(high[1:] - close[:-1])
    tr_6h3 = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.nan], np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))])
    atr_6h = np.full(n, np.nan)
    for i in range(6, len(tr_6h)):
        atr_6h[i] = np.mean(tr_6h[i-5:i+1])
    
    # Range filter: current 6h ATR < 1.5 * 14d ATR (avoid choppy markets)
    range_filter = (atr_6h < 1.5 * atr_14d_aligned) & (~np.isnan(atr_6h)) & (~np.isnan(atr_14d_aligned))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 6)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(range_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R1, with range filter (trending market)
            if close[i] > r1_w_aligned[i] and range_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1, with range filter (trending market)
            elif close[i] < s1_w_aligned[i] and range_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to weekly pivot or breaks below weekly S1
            if (not np.isnan(pivot_w_aligned[i]) and close[i] < pivot_w_aligned[i]) or \
               (not np.isnan(s1_w_aligned[i]) and close[i] < s1_w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly pivot or breaks above weekly R1
            if (not np.isnan(pivot_w_aligned[i]) and close[i] > pivot_w_aligned[i]) or \
               (not np.isnan(r1_w_aligned[i]) and close[i] > r1_w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_RangeFilter_v1"
timeframe = "6h"
leverage = 1.0