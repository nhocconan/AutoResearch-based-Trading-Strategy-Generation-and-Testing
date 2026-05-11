#!/usr/bin/env python3
name = "4h_Donchian20_Trend_Volume_1d"
timeframe = "4h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up_1d = close_1d > ema20_1d
    
    # Get 1d Donchian breakout levels (20-day high/low)
    high_20 = np.full(len(close_1d), np.nan)
    low_20 = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        high_20[i] = np.max(close_1d[i-20:i])
        low_20[i] = np.min(close_1d[i-20:i])
    
    # Get 4h Donchian breakout levels (20-period)
    high_20_4h = np.full(n, np.nan)
    low_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        high_20_4h[i] = np.max(high[i-20:i])
        low_20_4h[i] = np.min(low[i-20:i])
    
    # Align daily indicators to 4h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trend_up_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high + daily trend + volume confirmation
            if (high[i] > high_20_4h[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian low + daily downtrend + volume confirmation
            elif (low[i] < low_20_4h[i] and 
                  not trend_up_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low or daily trend changes
            if (low[i] < low_20_4h[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high or daily trend changes
            if (high[i] > high_20_4h[i] or trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals