#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Pivot point
    pivot = (high_prev + low_prev + close_prev) / 3
    # Camarilla levels
    S1 = close_prev - (high_prev - low_prev) * 1.0833
    S2 = close_prev - (high_prev - low_prev) * 1.2500
    S3 = close_prev - (high_prev - low_prev) * 1.5000
    R1 = close_prev + (high_prev - low_prev) * 1.0833
    R2 = close_prev + (high_prev - low_prev) * 1.2500
    R3 = close_prev + (high_prev - low_prev) * 1.5000
    
    # Align S3 and R3 to 4h (need extra delay for level confirmation)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3, additional_delay_bars=1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3, additional_delay_bars=1)
    
    # Daily EMA34 for trend filter
    daily_close = df_1d['close'].values
    ema34_d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_d)
    
    # Volume confirmation: 20-period volume average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(S3_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(ema34_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks below S3 (support) in uptrend with volume
            if (close[i] < S3_aligned[i] and 
                close[i] > ema34_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks above R3 (resistance) in downtrend with volume
            elif (close[i] > R3_aligned[i] and 
                  close[i] < ema34_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back above S3 or trend changes
            if (close[i] > S3_aligned[i] or close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back below R3 or trend changes
            if (close[i] < R3_aligned[i] or close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals