#!/usr/bin/env python3
name = "4h_1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily close for Camarilla calculation
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        # Use previous day's high, low, close
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Camarilla formula
        camarilla_r3[i] = prev_close + (prev_high - prev_low) * 1.1 / 4
        camarilla_s3[i] = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Daily EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema34
    
    # Align daily indicators to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and 
                  not trend_up_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend changes
            if (close[i] < camarilla_s3_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or trend changes
            if (close[i] > camarilla_r3_aligned[i] or trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals