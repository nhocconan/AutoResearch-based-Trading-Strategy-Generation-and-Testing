#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Uses 12h Donchian channel (20-period high/low) for breakout signals
- Long breakout: price > upper Donchian + volume > 1.5x 20-period avg + price > 1d EMA34 (uptrend)
- Short breakdown: price < lower Donchian + volume > 1.5x 20-period avg + price < 1d EMA34 (downtrend)
- Exit: price reverts to middle of Donchian channel (mean of upper/lower)
- Donchian breakouts capture institutional participation in both bull/bear markets
- Volume confirmation reduces false breakouts in low-participation moves
- 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
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
    
    # Load 12h data ONCE before loop for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    middle_12h = (upper_12h + lower_12h) / 2.0  # Exit level
    
    # Align Donchian levels to 12h timeframe (wait for 12h bar close)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle_12h)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for Donchian/volume, 34 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price > upper Donchian + volume spike + price > 1d EMA34 (uptrend)
            if volume_spike and close[i] > ema_34_aligned[i]:
                if close[i] > upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price < lower Donchian + volume spike + price < 1d EMA34 (downtrend)
            elif volume_spike and close[i] < ema_34_aligned[i]:
                if close[i] < lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to middle of Donchian channel
            if close[i] <= middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle of Donchian channel
            if close[i] >= middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0