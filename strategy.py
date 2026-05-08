#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian_WeeklyPivot_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly close)
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Daily trend filter for institutional bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 50-period EMA on daily close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) on 6h timeframe
    # Calculate highest high and lowest low over last 20 periods
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(ema_50_6h[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high AND above weekly R3 AND above daily EMA50 + volume
            if (close[i] > high_max[i] and
                close[i] > r3_6h[i] and
                close[i] > ema_50_6h[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND below weekly S3 AND below daily EMA50 + volume
            elif (close[i] < low_min[i] and
                  close[i] < s3_6h[i] and
                  close[i] < ema_50_6h[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price falls back below Donchian low or below weekly S3
            if (close[i] < low_min[i] or
                close[i] < s3_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price rises back above Donchian high or above weekly R3
            if (close[i] > high_max[i] or
                close[i] > r3_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals