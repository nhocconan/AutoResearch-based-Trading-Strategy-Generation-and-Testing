#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Long breakout: price > upper Donchian channel (20-period high) + volume > 1.5x 20-period avg + price > 1w EMA50 (uptrend)
- Short breakdown: price < lower Donchian channel (20-period low) + volume > 1.5x 20-period avg + price < 1w EMA50 (downtrend)
- Exit: price reverts to 20-period EMA (mean reversion to intermediate trend)
- 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades
- Volume confirmation reduces false breakouts in low-participation moves
- Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag on 1d timeframe
- Donchian channels provide clear structure that works across bull/bear regimes
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
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Exit condition: 20-period EMA
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume, 50 for 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price > upper Donchian + volume spike + price > 1w EMA50 (uptrend)
            if volume_spike and close[i] > ema_50_1w_aligned[i]:
                if close[i] > high_roll[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price < lower Donchian + volume spike + price < 1w EMA50 (downtrend)
            elif volume_spike and close[i] < ema_50_1w_aligned[i]:
                if close[i] < low_roll[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to 20-period EMA
            if close[i] <= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to 20-period EMA
            if close[i] >= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0