#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Uses 6h Donchian channel (20-period high/low) for breakout signals
- Long breakout: price > upper channel + volume > 1.5x 20-period avg + price > 12h EMA50 (uptrend)
- Short breakdown: price < lower channel + volume > 1.5x 20-period avg + price < 12h EMA50 (downtrend)
- Exit: price reverts to 12-period EMA of 6h close (mean reversion to intermediate trend)
- 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume confirmation reduces false breakouts in low-participation moves
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
- Donchian channels work in both bull (breakouts) and bear (breakdowns) markets
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
    
    # Donchian channel (20-period) on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Exit condition: 12-period EMA of 6h close (for mean reversion)
    ema_12_6h = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume, 50 for 12h EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_12_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price > upper Donchian + volume spike + price > 12h EMA50 (uptrend)
            if volume_spike and close[i] > ema_50_12h_aligned[i]:
                if close[i] > highest_20[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price < lower Donchian + volume spike + price < 12h EMA50 (downtrend)
            elif volume_spike and close[i] < ema_50_12h_aligned[i]:
                if close[i] < lowest_20[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to 12-period EMA of 6h close
            if close[i] <= ema_12_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to 12-period EMA of 6h close
            if close[i] >= ema_12_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0