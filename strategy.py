#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Fade_Trend
Hypothesis: Fade at R3/S3 levels of daily Camarilla pivots on 6h timeframe with 12h EMA trend filter.
In ranging markets, price often reverses at R3/S3 (strong support/resistance). In trending markets,
only take fades that align with the 12h trend (EMA50) to avoid counter-trend trades.
Designed to work in both bull and bear markets by combining mean reversion with trend filtering.
"""

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
    
    # Calculate 1-day Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels (R3, S3)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1 / 4
    s3_1d = pivot_1d - range_1d * 1.1 / 4
    
    # Align 1-day levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 12-hour EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    # Use pandas EMA for better performance and accuracy
    close_series = pd.Series(close_12h)
    ema50_12h = close_series.ewm(span=50, adjust=False, min_periods=50).values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA ready
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long fade: price at S3 with 12h uptrend
            if (close[i] <= s3_1d_aligned[i] and close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short fade: price at R3 with 12h downtrend
            elif (close[i] >= r3_1d_aligned[i] and close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back above S3 or trend changes
            if (close[i] > s3_1d_aligned[i] or close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back below R3 or trend changes
            if (close[i] < r3_1d_aligned[i] or close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Fade_Trend"
timeframe = "6h"
leverage = 1.0