#!/usr/bin/env python3
name = "6h_Weekly_Camarilla_Trend_Filter"
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
    
    # Get weekly data for trend and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up_1w = close_1w > ema20_1w
    
    # Weekly Camarilla levels (R3, S3)
    R3 = np.full(len(high_1w), np.nan)
    S3 = np.full(len(high_1w), np.nan)
    
    for i in range(1, len(high_1w)):
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        prev_close = close_1w[i-1]
        range_val = prev_high - prev_low
        if range_val > 0:
            R3[i] = prev_close + range_val * 1.1 / 4
            S3[i] = prev_close - range_val * 1.1 / 4
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Daily volume average for comparison
    vol_avg_1d = np.full(len(volume_1d), np.nan)
    for i in range(len(volume_1d)):
        if i < 10:
            if i > 0:
                vol_avg_1d[i] = np.mean(volume_1d[:i+1])
        else:
            vol_avg_1d[i] = np.mean(volume_1d[i-9:i+1])
    
    # Align all indicators to 6h timeframe
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Volume moving average (10-period) for 6h timeframe
    vol_ma10 = np.full(n, np.nan)
    for i in range(n):
        if i < 10:
            if i > 0:
                vol_ma10[i] = np.mean(volume[:i+1])
        else:
            vol_ma10[i] = np.mean(volume[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or
            np.isnan(trend_up_1w_aligned[i]) or
            np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(vol_ma10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + weekly uptrend + volume above daily average
            if (close[i] > R3_aligned[i] and 
                trend_up_1w_aligned[i] and 
                volume[i] > vol_ma10[i] and
                volume[i] > vol_avg_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + weekly downtrend + volume above daily average
            elif (close[i] < S3_aligned[i] and 
                  not trend_up_1w_aligned[i] and 
                  volume[i] > vol_ma10[i] and
                  volume[i] > vol_avg_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend changes to down
            if (close[i] < S3_aligned[i] or 
                not trend_up_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or trend changes to up
            if (close[i] > R3_aligned[i] or 
                trend_up_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals