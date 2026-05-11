#!/usr/bin/env python3
name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume_Filter"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA20 for trend
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up_4h = close_4h > ema20_4h
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    R1 = np.full(len(high_1d), np.nan)
    S1 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        if range_val > 0:
            R1[i] = prev_close + range_val * 1.1 / 12
            S1[i] = prev_close - range_val * 1.1 / 12
    
    # Align indicators to 1h timeframe
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume moving average (24-period) for confirmation
    vol_ma24 = np.full(n, np.nan)
    for i in range(n):
        if i < 24:
            if i > 0:
                vol_ma24[i] = np.mean(volume[:i+1])
        else:
            vol_ma24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or
            np.isnan(trend_up_4h_aligned[i]) or
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + uptrend + volume confirmation
            if (close[i] > R1_aligned[i] and 
                trend_up_4h_aligned[i] and 
                volume[i] > 1.5 * vol_ma24[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + downtrend + volume confirmation
            elif (close[i] < S1_aligned[i] and 
                  not trend_up_4h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma24[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend changes
            if (close[i] < S1_aligned[i] or 
                not trend_up_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 or trend changes
            if (close[i] > R1_aligned[i] or 
                trend_up_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals