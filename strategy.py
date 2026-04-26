#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_VolumeConfirmation
Hypothesis: 6-hour Donchian(20) breakout with 12-hour EMA50 trend filter and volume spike confirmation.
Enters long when price breaks above 20-bar high with bullish 12h trend and volume spike.
Enters short when price breaks below 20-bar low with bearish 12h trend and volume spike.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
Designed to work in both bull and bear markets by following the 12h trend direction only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-bar lookback + 50-period EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Calculate Donchian channels for current bar
        donchian_high = np.max(high[i-19:i+1])  # 20-bar high including current
        donchian_low = np.min(low[i-19:i+1])    # 20-bar low including current
        
        # Long logic: break above Donchian high + bullish 12h trend + volume spike
        if close[i] > donchian_high and close[i] > ema_50_12h_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below Donchian low + bearish 12h trend + volume spike
        elif close[i] < donchian_low and close[i] < ema_50_12h_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Donchian level
        elif position == 1 and close[i] < donchian_low:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_high:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0