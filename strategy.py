#!/usr/bin/env python3
name = "6h_WeeklyPivot_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly pivot calculation (using Friday close as weekly close)
    # Get daily data first to compute weekly
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly high/low/close from daily data
    weekly_high = np.full(len(df_1d), np.nan)
    weekly_low = np.full(len(df_1d), np.nan)
    weekly_close = np.full(len(df_1d), np.nan)
    
    # Simple weekly aggregation: every 5 days (approximation)
    for i in range(4, len(df_1d), 5):
        if i < len(df_1d):
            start_idx = max(0, i-4)
            weekly_high[i] = np.max(df_1d['high'].iloc[start_idx:i+1])
            weekly_low[i] = np.min(df_1d['low'].iloc[start_idx:i+1])
            weekly_close[i] = df_1d['close'].iloc[i]
    
    # Pivot points calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivots to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily trend filter: EMA(34)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above R3 with bullish daily trend and volume
            if close[i] > r3_6h[i] and close[i] > ema34_1d_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with bearish daily trend and volume
            elif close[i] < s3_6h[i] and close[i] < ema34_1d_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below pivot or trend reversal
            if close[i] < pivot_6h[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above pivot or trend reversal
            if close[i] > pivot_6h[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals