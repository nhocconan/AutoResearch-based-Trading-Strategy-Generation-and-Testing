#!/usr/bin/env python3
name = "6H_Weekly_Pivot_Donchian_Breakout_Trend_Filter"
timeframe = "6h"
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
    
    # Get weekly data for pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using standard formula)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    r1_w = pivot_w + (range_w * 1.1 / 12)
    s1_w = pivot_w - (range_w * 1.1 / 12)
    r2_w = pivot_w + (range_w * 1.1 / 6)
    s2_w = pivot_w - (range_w * 1.1 / 6)
    r3_w = pivot_w + (range_w * 1.1 / 4)
    s3_w = pivot_w - (range_w * 1.1 / 4)
    r4_w = pivot_w + (range_w * 1.1 / 2)
    s4_w = pivot_w - (range_w * 1.1 / 2)
    
    # Get daily data for Donchian channel and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # Donchian channel (20-day)
    upper_dc = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA50 for trend filter
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly pivot levels to 6h
    r1_w_aligned = align_htf_to_ltf(prices, df_weekly, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_weekly, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_weekly, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_weekly, s2_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_weekly, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_weekly, s3_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_weekly, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_weekly, s4_w)
    
    # Align daily indicators to 6h
    upper_dc_aligned = align_htf_to_ltf(prices, df_daily, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_daily, lower_dc)
    ema50_d_aligned = align_htf_to_ltf(prices, df_daily, ema50_d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(upper_dc_aligned[i]) or np.isnan(lower_dc_aligned[i]) or 
            np.isnan(ema50_d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 weekly pivot AND above daily Donchian upper AND above daily EMA50 AND volume confirmation
            if (close[i] > r3_w_aligned[i] and 
                close[i] > upper_dc_aligned[i] and 
                close[i] > ema50_d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 weekly pivot AND below daily Donchian lower AND below daily EMA50 AND volume confirmation
            elif (close[i] < s3_w_aligned[i] and 
                  close[i] < lower_dc_aligned[i] and 
                  close[i] < ema50_d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below daily Donchian lower OR below daily EMA50
            if close[i] < lower_dc_aligned[i] or close[i] < ema50_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above daily Donchian upper OR above daily EMA50
            if close[i] > upper_dc_aligned[i] or close[i] > ema50_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals