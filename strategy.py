#!/usr/bin/env python3
name = "6h_1d_1w_ElderRay_BullBear_Power_Trend_Volume"
timeframe = "6h"
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
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Elder Ray Power: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up = close_1w > ema20
    
    # Align indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    # Volume moving average (10-period) for confirmation
    vol_ma10 = np.zeros(n)
    for i in range(n):
        if i < 10:
            vol_ma10[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma10[i] = np.mean(volume[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or
            np.isnan(vol_ma10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power positive + Uptrend + Volume confirmation
            if (bull_power_aligned[i] > 0 and 
                trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma10[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative + Downtrend + Volume confirmation
            elif (bear_power_aligned[i] < 0 and 
                  not trend_up_aligned[i] and 
                  volume[i] > 1.5 * vol_ma10[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power negative or trend changes
            if (bear_power_aligned[i] < 0 or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power positive or trend changes
            if (bull_power_aligned[i] > 0 or trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals