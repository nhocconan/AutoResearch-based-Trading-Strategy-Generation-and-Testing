#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Long breakout: price > upper Donchian(20) + volume > 1.5x 20-period avg + price > 1d EMA34 (uptrend)
- Short breakdown: price < lower Donchian(20) + volume > 1.5x 20-period avg + price < 1d EMA34 (downtrend)
- Exit: price reverts to midpoint of Donchian channel or opposite breakout level
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
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
    
    # Donchian Channel (20-period) - calculated on primary timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (wait for daily close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for Donchian, 34 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or 
            np.isnan(donchian_mid[i]) or 
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
                if close[i] > high_roll[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price < lower Donchian + volume spike + price < 1d EMA34 (downtrend)
            elif volume_spike and close[i] < ema_34_aligned[i]:
                if close[i] < low_roll[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint or breaks below lower Donchian (failed breakout)
            if close[i] <= donchian_mid[i] or close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint or breaks above upper Donchian (failed breakdown)
            if close[i] >= donchian_mid[i] or close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0