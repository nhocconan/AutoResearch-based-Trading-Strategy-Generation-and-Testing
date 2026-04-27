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
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot from daily data (using previous week's data)
    # We'll calculate pivot points for each day based on the previous week
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high/low/close for pivot (using 5-day lookback for week)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot points
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Get 6h data for Donchian channel (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # 6h Donchian(20) channel
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Get 6h volume for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    upper_aligned = align_htf_to_ltf(prices, df_6h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_6h, lower)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need all indicators
    start_idx = max(20, 20, 20)  # Donchian(20), weekly pivot(5), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-period average
        vol_confirm = volume[i] > vol_ma_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation
            if close[i] > r3_aligned[i] and vol_confirm:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with volume confirmation
            elif close[i] < s3_aligned[i] and vol_confirm:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below R1 or reverses at R3
            if close[i] < r1_aligned[i] or (close[i] < r3_aligned[i] and i > start_idx and close[i-1] >= r3_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above S1 or reverses at S3
            if close[i] > s1_aligned[i] or (close[i] > s3_aligned[i] and i > start_idx and close[i-1] <= s3_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0