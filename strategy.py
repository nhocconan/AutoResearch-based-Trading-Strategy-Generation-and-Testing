#!/usr/bin/env python3
# 12H_1W_1D_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Use weekly trend filter (price > weekly SMA50) with weekly and daily
# Camarilla R3/S3 breakouts for stronger signals. Volume confirmation ensures
# breakout conviction. Weekly trend filter reduces whipsaws in bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12H_1W_1D_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Get weekly data for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly SMA50 trend filter
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Weekly Camarilla levels (R3, S3)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + range_1w * 1.1  # R3 = pivot + (range * 1.1)
    s3_1w = pivot_1w - range_1w * 1.1  # S3 = pivot - (range * 1.1)
    
    # Daily Camarilla levels (R3, S3)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1  # R3 = pivot + (range * 1.1)
    s3_1d = pivot_1d - range_1d * 1.1  # S3 = pivot - (range * 1.1)
    
    # Align all to 12h
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    volume_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(sma50_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or \
           np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly R3 AND daily R3 + above weekly SMA50 + volume confirmation
            if close[i] > r3_1w_aligned[i] and close[i] > r3_1d_aligned[i] and close[i] > sma50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S3 AND daily S3 + below weekly SMA50 + volume confirmation
            elif close[i] < s3_1w_aligned[i] and close[i] < s3_1d_aligned[i] and close[i] < sma50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly S3 (trend weakness)
            if close[i] < s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly R3 (trend strength)
            if close[i] > r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals