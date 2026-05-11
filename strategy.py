#!/usr/bin/env python3
name = "6h_ElderRay_1dTrend_Volume"
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
    
    # Get 1d data for Elder Ray (13-period EMA) and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    close_series = pd.Series(close_1d)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high_1d - ema13  # Bull power: high - EMA13
    bear_power = low_1d - ema13   # Bear power: low - EMA13
    
    # Trend filter: 50-period EMA on 1d
    ema50_1d = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Volume moving average (20-period) for confirmation on 6h
    vol_ma20 = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            if i > 0:
                vol_ma20[i] = np.mean(volume[:i+1])
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bull power > 0 + uptrend + volume confirmation
            if (bull_power_aligned[i] > 0 and 
                trend_up_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: bear power < 0 + downtrend + volume confirmation
            elif (bear_power_aligned[i] < 0 and 
                  not trend_up_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bull power <= 0 or trend changes
            if (bull_power_aligned[i] <= 0 or not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bear power >= 0 or trend changes
            if (bear_power_aligned[i] >= 0 or trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals