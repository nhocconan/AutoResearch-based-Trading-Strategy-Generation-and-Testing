#!/usr/bin/env python3
# 6h_1d_WeeklyPivot_R1S1_Breakout_Volume_TrendFilter_v1
# Hypothesis: Use 1d and 1w Pivot levels (R1/S1) with 6h breakout, volume confirmation, and 1d EMA34 trend filter.
# Only trade breakouts aligned with 1d trend. Weekly pivots provide stronger support/resistance for longer-term moves.
# Designed to work in both bull and bear markets by following trend and requiring volume confirmation.

name = "6h_1d_WeeklyPivot_R1S1_Breakout_Volume_TrendFilter_v1"
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
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d typical price and pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    pivot_1d = typical_price_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Calculate 1w typical price and pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    pivot_1w = typical_price_1w
    r1_1w = close_1w + (range_1w * 1.1 / 12)
    s1_1w = close_1w - (range_1w * 1.1 / 12)
    
    # Align 1d and 1w levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above both 1d R1 and 1w R1 with volume spike and uptrend
            if (close[i] > max(r1_1d_aligned[i], r1_1w_aligned[i]) * 1.005 and 
                volume[i] > 2.5 * volume_ma[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below both 1d S1 and 1w S1 with volume spike and downtrend
            elif (close[i] < min(s1_1d_aligned[i], s1_1w_aligned[i]) * 0.995 and 
                  volume[i] > 2.5 * volume_ma[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 1d S1 or trend reverses
            if close[i] < s1_1d_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 1d R1 or trend reverses
            if close[i] > r1_1d_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals