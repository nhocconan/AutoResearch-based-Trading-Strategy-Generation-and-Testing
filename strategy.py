#!/usr/bin/env python3
name = "6h_12h_1d_VolumeSpike_Breakout_Confluence"
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
    
    # Get 12h data for Donchian breakout structure
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels (20-period) on 12h
    upper_20 = np.full(len(high_12h), np.nan)
    lower_20 = np.full(len(high_12h), np.nan)
    
    for i in range(len(high_12h)):
        if i >= 19:
            upper_20[i] = np.max(high_12h[i-19:i+1])
            lower_20[i] = np.min(low_12h[i-19:i+1])
    
    # Get 1d data for volume spike and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma20 = np.full(len(volume_1d), np.nan)
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma20[i] = np.mean(volume_1d[i-19:i+1])
    
    volume_spike = np.zeros(len(volume_1d))
    for i in range(len(volume_1d)):
        if not np.isnan(vol_ma20[i]) and volume_1d[i] > 2.0 * vol_ma20[i]:
            volume_spike[i] = 1
    
    # Trend filter: EMA(34) on 1d close
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema34
    
    # Align all indicators to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(trend_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume spike + uptrend
            if (close[i] > upper_20_aligned[i] and 
                volume_spike_aligned[i] and 
                trend_up_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume spike + downtrend
            elif (close[i] < lower_20_aligned[i] and 
                  volume_spike_aligned[i] and 
                  not trend_up_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower or trend changes
            if (close[i] < lower_20_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper or trend changes
            if (close[i] > upper_20_aligned[i] or trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals