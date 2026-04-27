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
    
    # Get daily data for pivot points and volume analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily pivot points (standard)
    pivot_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    r1_1d = 2 * pivot_1d - low_1d[:-1]
    s1_1d = 2 * pivot_1d - high_1d[:-1]
    r2_1d = pivot_1d + (high_1d[:-1] - low_1d[:-1])
    s2_1d = pivot_1d - (high_1d[:-1] - low_1d[:-1])
    r3_1d = high_1d[:-1] + 2 * (pivot_1d - low_1d[:-1])
    s3_1d = low_1d[:-1] - 2 * (high_1d[:-1] - pivot_1d)
    r4_1d = r3_1d + (high_1d[:-1] - low_1d[:-1])
    s4_1d = s3_1d - (high_1d[:-1] - low_1d[:-1])
    
    # Align pivot levels to 6h timeframe (use previous day's pivots)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], pivot_1d]))
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], r1_1d]))
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], s1_1d]))
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], r2_1d]))
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], s2_1d]))
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], r3_1d]))
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], s3_1d]))
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], r4_1d]))
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], s4_1d]))
    
    # Calculate daily volume average (20-day)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get 12h data for trend filter: EMA(50) on 12h close
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h_50 = np.full(len(df_12h), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_12h)):
        if i < 49:
            ema_12h_50[i] = np.mean(close_12h[:i+1]) if i > 0 else close_12h[i]
        else:
            if np.isnan(ema_12h_50[i-1]):
                ema_12h_50[i] = np.mean(close_12h[i-49:i+1])
            else:
                ema_12h_50[i] = close_12h[i] * alpha + ema_12h_50[i-1] * (1 - alpha)
    
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(ema_12h_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20_1d_aligned[i] if vol_ma_20_1d_aligned[i] > 0 else 1.0
        
        if position == 0:
            # Long: Price breaks above R3 with volume, in uptrend
            if (price > r3_1d_aligned[i] and 
                vol_ratio > 1.5 and 
                ema_12h_50_aligned[i] > ema_12h_50_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume, in downtrend
            elif (price < s3_1d_aligned[i] and 
                  vol_ratio > 1.5 and 
                  ema_12h_50_aligned[i] < ema_12h_50_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price falls below R2 or trend turns down
            if (price < r2_1d_aligned[i] or 
                ema_12h_50_aligned[i] < ema_12h_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price rises above S2 or trend turns up
            if (price > s2_1d_aligned[i] or 
                ema_12h_50_aligned[i] > ema_12h_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_PivotBreakout_R3S3_12hEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0