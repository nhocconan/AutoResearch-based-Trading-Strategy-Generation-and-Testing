#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: Donchian(20) breakout on 12h with 1d EMA34 trend filter and volume spike confirmation.
Enters long when price breaks above 20-bar high with bullish 1d trend and volume spike.
Enters short when price breaks below 20-bar low with bearish 1d trend and volume spike.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 50-150 total trades over 4 years.
Works in both bull and bear markets by following the 1d trend direction only.
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
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) channels on 12h
    # Need to calculate on 12h then align to 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 20-period rolling max/min on 12h
    df_high = pd.Series(high_12h)
    df_low = pd.Series(low_12h)
    donchian_high_20 = df_high.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = df_low.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    dh_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period Donchian + EMA34)
    start_idx = 34 + 20  # EMA34 warmup + Donchian20 warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dh_20_aligned[i]) or np.isnan(dl_20_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above Donchian high + bullish 1d trend + volume spike
        if close[i] > dh_20_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below Donchian low + bearish 1d trend + volume spike
        elif close[i] < dl_20_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Donchian level
        elif position == 1 and close[i] < dl_20_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > dh_20_aligned[i]:
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

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0