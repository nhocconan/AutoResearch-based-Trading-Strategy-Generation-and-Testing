#!/usr/bin/env python3
name = "4h_Donchian_20_Volume_Spike"
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
    
    # Donchian channel (20-period)
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:  # 20 bars: i-19 to i inclusive
            highest_20[i] = np.max(high[i-19:i+1])
            lowest_20[i] = np.min(low[i-19:i+1])
    
    # Volume spike filter: volume > 2.0 * 20-period average
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian and volume data
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or
            np.isnan(vol_avg_20[i]) or
            np.isnan(trend_up_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions
        volume_spike = volume[i] > 2.0 * vol_avg_20[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + uptrend + volume spike
            if (close[i] > highest_20[i] and 
                trend_up_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + downtrend + volume spike
            elif (close[i] < lowest_20[i] and 
                  not trend_up_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian or trend changes
            if (close[i] < lowest_20[i] or not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian or trend changes
            if (close[i] > highest_20[i] or trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals