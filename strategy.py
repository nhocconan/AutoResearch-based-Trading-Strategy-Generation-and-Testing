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
    
    # Get daily data for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (previous day)
    pivot_1d = np.full(len(df_1d), np.nan)
    r1_1d = np.full(len(df_1d), np.nan)
    s1_1d = np.full(len(df_1d), np.nan)
    r2_1d = np.full(len(df_1d), np.nan)
    s2_1d = np.full(len(df_1d), np.nan)
    r3_1d = np.full(len(df_1d), np.nan)
    s3_1d = np.full(len(df_1d), np.nan)
    r4_1d = np.full(len(df_1d), np.nan)
    s4_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        pp = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        pivot_1d[i] = pp
        r1_1d[i] = 2 * pp - low_1d[i-1]
        s1_1d[i] = 2 * pp - high_1d[i-1]
        r2_1d[i] = pp + (high_1d[i-1] - low_1d[i-1])
        s2_1d[i] = pp - (high_1d[i-1] - low_1d[i-1])
        r3_1d[i] = high_1d[i-1] + 2 * (pp - low_1d[i-1])
        s3_1d[i] = low_1d[i-1] - 2 * (high_1d[i-1] - pp)
        r4_1d[i] = pp + 3 * (high_1d[i-1] - low_1d[i-1])
        s4_1d[i] = pp - 3 * (high_1d[i-1] - low_1d[i-1])
    
    # Align pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = np.full(len(df_1d), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_1d)):
        if i < 33:
            ema_34_1d[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_34_1d[i-1]):
                ema_34_1d[i] = np.mean(close_1d[i-33:i+1])
            else:
                ema_34_1d[i] = close_1d[i] * alpha + ema_34_1d[i-1] * (1 - alpha)
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike detector (volume > 2x 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    volume_spike = np.full(n, np.nan)
    volume_spike[20:] = volume[20:] > (2 * vol_ma_20[20:])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: price breaks above R4 with volume spike + daily uptrend
            if (price > r4_1d_aligned[i] and 
                volume_spike[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short setup: price breaks below S4 with volume spike + daily downtrend
            elif (price < s4_1d_aligned[i] and 
                  volume_spike[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below R3 or trend turns down
            if (price < r3_1d_aligned[i] or 
                ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above S3 or trend turns up
            if (price > s3_1d_aligned[i] or 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0