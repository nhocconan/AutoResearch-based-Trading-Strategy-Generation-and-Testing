#!/usr/bin/env python3
name = "1d_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "1d"
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
    
    # Get daily data for calculations (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (R3, S3)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    range_val = prev_high - prev_low
    
    # R3 = prev_close + (range * 1.1 / 4), S3 = prev_close - (range * 1.1 / 4)
    R3 = prev_close + range_val * 1.1 / 4
    S3 = prev_close - range_val * 1.1 / 4
    
    # Set first day to NaN (no previous day)
    R3[0] = np.nan
    S3[0] = np.nan
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up = close_1w > ema20
    
    # Align daily indicators to 1d timeframe (they are already daily)
    # For 1d timeframe, we use previous day's values for today's signal
    R3_today = np.roll(R3, 1)  # Yesterday's R3 for today's signal
    S3_today = np.roll(S3, 1)  # Yesterday's S3 for today's signal
    R3_today[0] = np.nan
    S3_today[0] = np.nan
    
    # Align weekly trend to daily
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R3_today[i]) or 
            np.isnan(S3_today[i]) or
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
            if (close[i] > R3_today[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + volume confirmation
            elif (close[i] < S3_today[i] and 
                  not trend_up_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend changes
            if (close[i] < S3_today[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or trend changes
            if (close[i] > R3_today[i] or trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals