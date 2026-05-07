#!/usr/bin/env python3
name = "12H_Camarilla_R3_S3_Breakout_1wTrend_WeeklyVolume"
timeframe = "12h"
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
    
    # Get weekly data for trend filter and weekly volume for confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly volume average for confirmation
    volume_1w = df_1w['volume'].values
    volume_avg_1w = pd.Series(volume_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_avg_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = close + (high - low) * 1.1 / 4, S3 = close - (high - low) * 1.1 / 4
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_avg_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly volume confirmation: current 12h volume > weekly average volume
        volume_confirmation = volume[i] > volume_avg_1w_aligned[i]
        
        if position == 0:
            # Long: price above weekly EMA20 (uptrend), 12h close above daily R3, volume confirmation
            if (close[i] > ema_20_1w_aligned[i] and 
                close[i] > r3_aligned[i] and 
                volume_confirmation):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA20 (downtrend), 12h close below daily S3, volume confirmation
            elif (close[i] < ema_20_1w_aligned[i] and 
                  close[i] < s3_aligned[i] and 
                  volume_confirmation):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly EMA20 (trend change)
            if close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly EMA20 (trend change)
            if close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals