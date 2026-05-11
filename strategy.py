#!/usr/bin/env python3
name = "6h_Fisher_Transform_Regime_12hVolumeSpike"
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
    
    # 12h Fisher Transform (9-period) for mean reversion signals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 9:
        return np.zeros(n)
    hlc3_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3
    max_hlc3 = pd.Series(hlc3_12h).rolling(window=9, min_periods=9).max().values
    min_hlc3 = pd.Series(hlc3_12h).rolling(window=9, min_periods=9).min().values
    value1 = np.where(max_hlc3 - min_hlc3 != 0, 
                      0.33 * 2 * ((hlc3_12h - min_hlc3) / (max_hlc3 - min_hlc3) - 0.5), 
                      0)
    value1 = np.clip(value1, -0.999, 0.999)
    fish = np.zeros_like(hlc3_12h)
    for i in range(1, len(hlc3_12h)):
        fish[i] = 0.5 * np.log((1 + value1[i]) / (1 - value1[i])) + 0.5 * fish[i-1]
    fish = np.clip(fish, -2, 2)
    fish_aligned = align_htf_to_ltf(prices, df_12h, fish)
    
    # 1d volume filter: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.5 * vol_ma20_1d_aligned
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for Fisher and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(fish_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 from below + volume filter
            if fish_aligned[i] > -1.5 and fish_aligned[i-1] <= -1.5 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 from above + volume filter
            elif fish_aligned[i] < 1.5 and fish_aligned[i-1] >= 1.5 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Fisher crosses below 0 or volume filter fails
            if fish_aligned[i] < 0 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Fisher crosses above 0 or volume filter fails
            if fish_aligned[i] > 0 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals