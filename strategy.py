#!/usr/bin/env python3
name = "4h_Weekly_Camarilla_R1_S1_Breakout_Trend_Volume"
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
    
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend
    weekly_close = df_1w['close'].values
    ema50_w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_trend = weekly_close > ema50_w
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous daily bar
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    R1 = close_prev + 1.1 * (high_prev - low_prev) / 12
    S1 = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    # Align to 4h
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation
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
        if (np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + price breaks above R1 + volume confirmation
            if (weekly_trend_aligned[i] and 
                close[i] > R1_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.30
                position = 1
            # Short: weekly downtrend + price breaks below S1 + volume confirmation
            elif (not weekly_trend_aligned[i] and 
                  close[i] < S1_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: weekly trend changes or price breaks below S1
            if (not weekly_trend_aligned[i] or close[i] < S1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: weekly trend changes or price breaks above R1
            if (weekly_trend_aligned[i] or close[i] > R1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals