#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_VolumeSpike_HTF
Hypothesis: 6h Donchian(20) breakout in direction of 1d EMA50 trend with volume confirmation.
Only trade breakouts aligned with higher timeframe trend to avoid counter-trend whipsaws.
Volume spike ensures institutional participation. Works in bull (trend continuation) and bear (trend persistence) markets.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for all indicators
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        
        if position == 0:
            # Flat - look for breakout entry in direction of 1d trend
            uptrend = close_val > ema_50_aligned[i]
            downtrend = close_val < ema_50_aligned[i]
            
            long_breakout = (close_val > highest_20[i]) and volume_spike and uptrend
            short_breakout = (close_val < lowest_20[i]) and volume_spike and downtrend
            
            if long_breakout:
                signals[i] = size
                position = 1
            elif short_breakout:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long - exit when price retraces to midpoint of Donchian channel
            midpoint = (highest_20[i] + lowest_20[i]) / 2.0
            if close_val < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price retraces to midpoint of Donchian channel
            midpoint = (highest_20[i] + lowest_20[i]) / 2.0
            if close_val > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_VolumeSpike_HTF"
timeframe = "6h"
leverage = 1.0