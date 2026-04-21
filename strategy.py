#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h Trend Filter + Volume Confirmation
Hypothesis: Donchian breakouts capture momentum bursts. 12h EMA trend filter ensures alignment with higher timeframe direction, reducing false signals in chop. Volume confirmation adds conviction. Works in bull (breakouts up) and bear (breakouts down) by being direction-agnostic with trend filter. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA34 for trend
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ok = vol_current > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price above Donchian high, above 12h EMA, volume confirmation
            if price > high_max[i] and price > ema_12h_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price below Donchian low, below 12h EMA, volume confirmation
            elif price < low_min[i] and price < ema_12h_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or reverses below 12h EMA
            if price < low_min[i] or price < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or reverses above 12h EMA
            if price > high_max[i] or price > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0