#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Long breakout: price > upper Donchian(20) + volume > 1.8x 20-period avg + price > 12h EMA50 (uptrend)
- Short breakdown: price < lower Donchian(20) + volume > 1.8x 20-period avg + price < 12h EMA50 (downtrend)
- Exit: price reverts to 20-period EMA (middle of channel)
- Uses 12h EMA50 for trend alignment to avoid counter-trend trades and capture medium-term momentum
- Volume confirmation filters low-participation breakouts
- Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag on 4h timeframe
- Donchian channels provide objective volatility-based structure that adapts to changing market conditions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.8x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian Channel (20-period)
    upper_donch = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_donch = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Exit condition: 20-period EMA (middle of channel)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume/EMA20, 50 for 12h EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(upper_donch[i]) or 
            np.isnan(lower_donch[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price > upper Donchian + volume spike + price > 12h EMA50 (uptrend)
            if volume_spike and close[i] > ema_50_aligned[i]:
                if close[i] > upper_donch[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price < lower Donchian + volume spike + price < 12h EMA50 (downtrend)
            elif volume_spike and close[i] < ema_50_aligned[i]:
                if close[i] < lower_donch[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to 20-period EMA (middle of channel)
            if close[i] <= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to 20-period EMA (middle of channel)
            if close[i] >= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0