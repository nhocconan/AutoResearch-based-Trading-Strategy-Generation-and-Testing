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
    
    # Get daily data for pivot points and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    r4_1d = r3_1d + (high_1d - low_1d)
    s4_1d = s3_1d - (high_1d - low_1d)
    
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
    
    # Calculate 6h Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Calculate volume ratio: current volume / 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.full(n, np.nan)
    valid_vol = vol_ma > 0
    vol_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume confirmation: above average volume
        vol_confirmed = vol_ratio[i] > 1.2
        
        if position == 0:
            # Long: price breaks above R3 with volume + daily uptrend
            if (price > r3_1d_aligned[i] and 
                vol_confirmed and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume + daily downtrend
            elif (price < s3_1d_aligned[i] and 
                  vol_confirmed and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below R1 or daily trend turns down
            if (price < r1_1d_aligned[i] or 
                ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above S1 or daily trend turns up
            if (price > s1_1d_aligned[i] or 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_PivotBreakout_R3S3_Volume_DailyEMA34_Trend_v1"
timeframe = "6h"
leverage = 1.0