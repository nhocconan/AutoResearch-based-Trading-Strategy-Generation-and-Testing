#!/usr/bin/env python3
# 4h_Donchian_Breakout_20_VolumeSpike_TrendFilter
# Hypothesis: Donchian channel breakout with volume spike confirmation and 1-day trend filter.
# Goes long on breakout above upper band with volume spike and 1-day uptrend.
# Goes short on breakdown below lower band with volume spike and 1-day downtrend.
# Uses Donchian(20) for price channel, volume spike >2x average volume, and 1-day EMA(50) for trend.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4h_Donchian_Breakout_20_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for spike detection (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Warmup for Donchian and 1d EMA
    
    for i in range(start_idx, n):
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume spike condition (>2x average volume)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long entry: breakout above upper Donchian band + volume spike + 1-day uptrend
            if high[i] > high_max[i-1] and volume_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: breakdown below lower Donchian band + volume spike + 1-day downtrend
            elif low[i] < low_min[i-1] and volume_spike and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: breakdown below lower Donchian band OR trend reversal
            if low[i] < low_min[i-1] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: breakout above upper Donchian band OR trend reversal
            if high[i] > high_max[i-1] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals