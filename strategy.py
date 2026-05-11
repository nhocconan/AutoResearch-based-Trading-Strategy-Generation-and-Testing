#!/usr/bin/env python3
name = "12h_1w_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous weekly candle
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla R3, S3 levels (from previous week)
    camarilla_r3 = np.zeros(len(high_1w))
    camarilla_s3 = np.zeros(len(high_1w))
    
    for i in range(len(high_1w)):
        if i == 0:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
        else:
            # Using previous week's data
            ph = high_1w[i-1]
            pl = low_1w[i-1]
            pc = close_1w[i-1]
            camarilla_r3[i] = pc + (ph - pl) * 1.1 / 4  # R3
            camarilla_s3[i] = pc - (ph - pl) * 1.1 / 4  # S3
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1d > ema50
    
    # Align weekly Camarilla levels to 12h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Align daily trend to 12h
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
    
    start_idx = max(100, 20)
    
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
            # Long: price breaks above Camarilla R3 + uptrend + volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + downtrend + volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and 
                  not trend_up_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S3 or trend changes
            if (close[i] < camarilla_s3_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R3 or trend changes
            if (close[i] > camarilla_r3_aligned[i] or trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals