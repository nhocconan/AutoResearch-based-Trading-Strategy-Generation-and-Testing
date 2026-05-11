#!/usr/bin/env python3
name = "6h_Fisher_Transform_1dTrend_Filter"
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
    
    # Fisher Transform on 6h close (period=9)
    lookback = 9
    highest = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(close).rolling(window=lookback, min_periods=lookback).min().values
    range_hl = highest - lowest
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)  # avoid division by zero
    
    # Normalize price to [-1, 1]
    value = 2 * ((close - lowest) / range_hl - 0.5)
    value = np.clip(value, -0.999, 0.999)
    
    # Fisher Transform
    fish = np.zeros(n)
    fish[0] = 0
    for i in range(1, n):
        fish[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * fish[i-1]
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    
    # Align 1d trend to 6h timeframe
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Volume confirmation (20-period MA on 6h)
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(fish[i]) or 
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 + uptrend + volume confirmation
            if (fish[i] > -1.5 and fish[i-1] <= -1.5 and 
                trend_up_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 + downtrend + volume confirmation
            elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and 
                  not trend_up_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Fisher crosses below -1.5 or trend changes
            if (fish[i] < -1.5 and fish[i-1] >= -1.5) or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Fisher crosses above +1.5 or trend changes
            if (fish[i] > 1.5 and fish[i-1] <= 1.5) or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals