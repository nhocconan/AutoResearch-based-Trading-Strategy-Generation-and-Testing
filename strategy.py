#!/usr/bin/env python3
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
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 20-period Donchian channels
    upper_12h = np.full(len(high_12h), np.nan)
    lower_12h = np.full(len(low_12h), np.nan)
    for i in range(20-1, len(high_12h)):
        upper_12h[i] = np.max(high_12h[i-20+1:i+1])
        lower_12h[i] = np.min(low_12h[i-20+1:i+1])
    
    # Calculate 12h EMA200 for trend filter
    ema_period = 200
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier + ema_12h[i-1] * (1 - multiplier))
    
    # Get 1d data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot levels from daily OHLC (using last 5 days)
    pivot_points = np.full(len(high_1d), np.nan)
    r3_levels = np.full(len(high_1d), np.nan)
    s3_levels = np.full(len(high_1d), np.nan)
    r4_levels = np.full(len(high_1d), np.nan)
    s4_levels = np.full(len(high_1d), np.nan)
    
    for i in range(4, len(high_1d)):
        # Use last 5 days of data for weekly pivot
        hh = np.max(high_1d[i-4:i+1])
        ll = np.min(low_1d[i-4:i+1])
        pc = close_1d[i]
        
        pivot = (hh + ll + pc) / 3
        r1 = 2 * pivot - ll
        s1 = 2 * pivot - hh
        r2 = pivot + (hh - ll)
        s2 = pivot - (hh - ll)
        r3 = hh + 2 * (pivot - ll)
        s3 = ll - 2 * (hh - pivot)
        r4 = hh + 3 * (pivot - ll)
        s4 = ll - 3 * (hh - pivot)
        
        pivot_points[i] = pivot
        r3_levels[i] = r3
        s3_levels[i] = s3
        r4_levels[i] = r4
        s4_levels[i] = s4
    
    # Align indicators to 6h timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    pivot_points_aligned = align_htf_to_ltf(prices, df_1d, pivot_points)
    r3_levels_aligned = align_htf_to_ltf(prices, df_1d, r3_levels)
    s3_levels_aligned = align_htf_to_ltf(prices, df_1d, s3_levels)
    r4_levels_aligned = align_htf_to_ltf(prices, df_1d, r4_levels)
    s4_levels_aligned = align_htf_to_ltf(prices, df_1d, s4_levels)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, EMA200, pivot levels, and volume MA
    start_idx = max(20, 200, vol_period) + 10
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(pivot_points_aligned[i]) or
            np.isnan(r3_levels_aligned[i]) or np.isnan(s3_levels_aligned[i]) or
            np.isnan(r4_levels_aligned[i]) or np.isnan(s4_levels_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Breakout above Donchian upper + price > EMA200 + volume spike + above weekly R3
            if (price > upper_12h_aligned[i] and 
                price > ema_12h_aligned[i] and 
                vol_ratio > 1.5 and 
                price > r3_levels_aligned[i]):
                signals[i] = size
                position = 1
            # Short: Breakdown below Donchian lower + price < EMA200 + volume spike + below weekly S3
            elif (price < lower_12h_aligned[i] and 
                  price < ema_12h_aligned[i] and 
                  vol_ratio > 1.5 and 
                  price < s3_levels_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price falls below Donchian lower OR below weekly S3
            if (price < lower_12h_aligned[i] or 
                price < s3_levels_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price rises above Donchian upper OR above weekly R3
            if (price > upper_12h_aligned[i] or 
                price > r3_levels_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0