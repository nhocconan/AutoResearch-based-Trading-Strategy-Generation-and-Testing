#!/usr/bin/env python3
# Hypothesis: 6h Donchian breakout with weekly pivot bias and volume confirmation.
# Weekly pivot provides directional bias: price above weekly pivot = bullish bias (look for long breakouts),
# price below weekly pivot = bearish bias (look for short breakouts). Donchian(20) breakouts capture
# momentum in the direction of bias. Volume confirms breakout strength. Designed for 6h to target
# 50-150 total trades over 4 years (12-37/year). Works in bull/bear by using pivot for directional filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    r3 = high_w + 2 * (pivot - low_w)
    s3 = low_w - 2 * (high_w - pivot)
    
    # Align weekly pivot to 6h (use prior week's values for bias)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Donchian(20) channels on 6h
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: volume > 1.8x 20-period average (strict to reduce trades)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(highest[i]) or 
            np.isnan(lowest[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly bias: price above pivot = bullish bias, below = bearish bias
        bullish_bias = close[i] > pivot_aligned[i]
        bearish_bias = close[i] < pivot_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest[i-1]  # Break above prior period high
        breakout_down = close[i] < lowest[i-1]  # Break below prior period low
        
        # Entry conditions with bias and volume
        long_entry = bullish_bias and breakout_up and volume_filter[i]
        short_entry = bearish_bias and breakout_down and volume_filter[i]
        
        # Exit when price returns to pivot (mean reversion to bias level)
        long_exit = position == 1 and close[i] <= pivot_aligned[i]
        short_exit = position == -1 and close[i] >= pivot_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian_WeeklyPivotBias_Volume"
timeframe = "6h"
leverage = 1.0