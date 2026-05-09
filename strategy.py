#!/usr/bin/env python3
name = "6h_Fisher_Transform_1dTrend_Volume"
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
    
    # Get daily data for 1D trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Fisher Transform (9-period) on 6H prices
    hl2 = (high + low) / 2
    max_hl2 = pd.Series(hl2).rolling(window=9, min_periods=9).max().values
    min_hl2 = pd.Series(hl2).rolling(window=9, min_periods=9).min().values
    value = np.where(max_hl2 - min_hl2 != 0,
                     2 * ((hl2 - min_hl2) / (max_hl2 - min_hl2) - 0.5),
                     0)
    value = np.clip(value, -0.999, 0.999)
    fisher = np.zeros_like(hl2)
    fisher[0] = 0
    for i in range(1, len(hl2)):
        fisher[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * fisher[i-1]
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(9, 20, 100)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema100_1d_aligned[i]) or np.isnan(vol_ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above 1D EMA100
        uptrend = close[i] > ema100_1d_aligned[i]
        # Downtrend: price below 1D EMA100
        downtrend = close[i] < ema100_1d_aligned[i]
        # Volume surge
        volume_surge = volume[i] > vol_ema20[i] * 1.5
        
        if position == 0:
            # Enter long: Fisher crosses above -1.5 + uptrend + volume surge
            if fisher[i] > -1.5 and fisher[i-1] <= -1.5 and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Fisher crosses below +1.5 + downtrend + volume surge
            elif fisher[i] < 1.5 and fisher[i-1] >= 1.5 and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Fisher crosses below +1.5 OR trend turns down
            if fisher[i] < 1.5 and fisher[i-1] >= 1.5 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Fisher crosses above -1.5 OR trend turns up
            if fisher[i] > -1.5 and fisher[i-1] <= -1.5 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals