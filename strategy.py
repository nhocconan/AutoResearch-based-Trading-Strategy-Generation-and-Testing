#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period avg + price > 12h EMA50 (uptrend)
- Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period avg + price < 12h EMA50 (downtrend)
- Exit: price reverts to 20-period Donchian midpoint (mean of high and low channels)
- 12h EMA50 ensures alignment with intermediate trend to avoid counter-trend trades
- Volume confirmation reduces false breakouts in low-participation moves
- Target: 20-50 trades/year (75-200 total over 4 years) to minimize fee drag on 4h timeframe
- Donchian channels provide clear structure that works across bull/bear regimes via breakouts
"""

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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 20-period Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume, 50 for 12h EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price > Donchian high + volume spike + price > 12h EMA50 (uptrend)
            if volume_spike and close[i] > ema_50_aligned[i]:
                if close[i] > high_roll[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price < Donchian low + volume spike + price < 12h EMA50 (downtrend)
            elif volume_spike and close[i] < ema_50_aligned[i]:
                if close[i] < low_roll[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0