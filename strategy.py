#!/usr/bin/env python3
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
    
    # Get 1d data for weekly pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d data
    # Need to group daily data into weeks
    weeks_high = []
    weeks_low = []
    weeks_close = []
    
    # Simple approach: use last 5 days as proxy for weekly
    # More accurate would require proper week grouping, but this avoids look-ahead
    high_5d = np.full(len(df_1d), np.nan)
    low_5d = np.full(len(df_1d), np.nan)
    close_5d = np.full(len(df_1d), np.nan)
    
    for i in range(4, len(df_1d)):
        high_5d[i] = np.max(df_1d['high'].values[i-4:i+1])
        low_5d[i] = np.min(df_1d['low'].values[i-4:i+1])
        close_5d[i] = df_1d['close'].values[i]
    
    # Calculate pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    
    pivot = np.full(len(df_1d), np.nan)
    r1 = np.full(len(df_1d), np.nan)
    s1 = np.full(len(df_1d), np.nan)
    r2 = np.full(len(df_1d), np.nan)
    s2 = np.full(len(df_1d), np.nan)
    r3 = np.full(len(df_1d), np.nan)
    s3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if not (np.isnan(high_5d[i]) or np.isnan(low_5d[i]) or np.isnan(close_5d[i])):
            pivot[i] = (high_5d[i] + low_5d[i] + close_5d[i]) / 3.0
            r1[i] = 2 * pivot[i] - low_5d[i]
            s1[i] = 2 * pivot[i] - high_5d[i]
            r2[i] = pivot[i] + (high_5d[i] - low_5d[i])
            s2[i] = pivot[i] - (high_5d[i] - low_5d[i])
            r3[i] = high_5d[i] + 2 * (pivot[i] - low_5d[i])
            s3[i] = low_5d[i] - 2 * (high_5d[i] - pivot[i])
    
    # Get 60-period EMA for trend filter (using 1d close)
    ema_period = 60
    ema_1d = np.full(len(df_1d), np.nan)
    close_1d = df_1d['close'].values
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                        ema_1d[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align all indicators to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup
    start_idx = max(vol_period, 1) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(ema_1d_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: price breaks above R2 with volume, above EMA (bullish)
            if (price > r2_6h[i] and 
                vol_ratio > 1.5 and 
                price > ema_1d_6h[i]):
                signals[i] = size
                position = 1
            # Short: price breaks below S2 with volume, below EMA (bearish)
            elif (price < s2_6h[i] and 
                  vol_ratio > 1.5 and 
                  price < ema_1d_6h[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price falls below R1 or R2 (taking profit) or reverses below S1
            if (price < r1_6h[i] or 
                price < s1_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price rises above S1 or S2 (taking profit) or reverses above R1
            if (price > s1_6h[i] or 
                price > r1_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_PivotPoints_R2S2_Breakout_Volume_EMA60"
timeframe = "6h"
leverage = 1.0