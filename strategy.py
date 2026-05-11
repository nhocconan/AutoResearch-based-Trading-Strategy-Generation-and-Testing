#!/usr/bin/env python3
name = "4h_Weekly_Camarilla_Breakout_Trend_Volume"
timeframe = "4h"
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
    
    # Get weekly and daily data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 5 or len(df_1d) < 34:
        return np.zeros(n)
    
    # Weekly Camarilla pivot points (using previous week)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = pivot + (prev_week_high - prev_week_low) * 1.1 / 12
    r2 = pivot + (prev_week_high - prev_week_low) * 1.1 / 6
    r3 = pivot + (prev_week_high - prev_week_low) * 1.1 / 4
    s1 = pivot - (prev_week_high - prev_week_low) * 1.1 / 12
    s2 = pivot - (prev_week_high - prev_week_low) * 1.1 / 6
    s3 = pivot - (prev_week_high - prev_week_low) * 1.1 / 4
    
    # Daily trend filter (EMA34)
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d > np.roll(ema34_1d, 1))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d < np.roll(ema34_1d, 1))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 in weekly uptrend with volume surge
            if (close[i] > r3_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 in weekly downtrend with volume surge
            elif (close[i] < s3_aligned[i] and 
                  trend_down_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price falls below R1 or weekly trend changes
            if (close[i] < r1_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price rises above S1 or weekly trend changes
            if (close[i] > s1_aligned[i] or not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals