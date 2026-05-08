#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Camarilla_R3_S3_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume filter
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Use previous week's data to avoid look-ahead
    prev_high_1w = np.roll(df_1w['high'].values, 1)
    prev_low_1w = np.roll(df_1w['low'].values, 1)
    prev_close_1w = np.roll(df_1w['close'].values, 1)
    prev_high_1w[0] = df_1w['high'].values[0]
    prev_low_1w[0] = df_1w['low'].values[0]
    prev_close_1w[0] = df_1w['close'].values[0]
    
    pivot_w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    range_w = prev_high_1w - prev_low_1w
    r3 = pivot_w + (range_w * 1.1)  # R3 at 2.1 pivot
    s3 = pivot_w - (range_w * 1.1)  # S3 at -1.1 pivot
    
    # Align weekly levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and weekly pivot
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R3, price above 1d EMA50, volume above average
            long_cond = (close[i] > r3_aligned[i] and 
                        close[i] > ema50_1d_aligned[i] and
                        volume[i] > vol_ma20_1d_aligned[i])
            
            # Short: Price breaks below weekly S3, price below 1d EMA50, volume above average
            short_cond = (close[i] < s3_aligned[i] and 
                         close[i] < ema50_1d_aligned[i] and
                         volume[i] > vol_ma20_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below weekly S3 OR price crosses below 1d EMA50
            if close[i] < s3_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above weekly R3 OR price crosses above 1d EMA50
            if close[i] > r3_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals