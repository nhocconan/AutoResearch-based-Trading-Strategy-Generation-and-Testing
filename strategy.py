#!/usr/bin/env python3
name = "1d_Camarilla_Pivot_Breakout_WeeklyTrend_Volume"
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
    
    # Get 1w data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get 1d data for Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current day volume > 20-period average volume
    vol_1d = df_1d['volume'].values
    vol_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter = vol_1d > vol_avg
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly EMA20 (uptrend), daily close above daily R3, volume confirmation
            if (close[i] > ema_20_1w_aligned[i] and 
                close[i] > r3_aligned[i] and 
                volume_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA20 (downtrend), daily close below daily S3, volume confirmation
            elif (close[i] < ema_20_1w_aligned[i] and 
                  close[i] < s3_aligned[i] and 
                  volume_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below daily S3 (strong reversal)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above daily R3 (strong reversal)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals