#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeFilter
Hypothesis: 12-hour Donchian(20) breakout with 1-week EMA50 trend filter and volume spike confirmation.
Enters long when price breaks above 20-period high with bullish weekly trend and volume spike.
Enters short when price breaks below 20-period low with bearish weekly trend and volume spike.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 50-150 total trades over 4 years.
Works in both bull and bear markets by following the weekly trend direction only.
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
    
    # Calculate Donchian(20) on 12h timeframe (using completed bars only)
    df_12h = get_htf_data(prices, '12h')
    
    # Rolling high/low on completed 12h bars (shifted by 1 to avoid look-ahead)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Prior completed 12h bar's rolling window
    roll_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    roll_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed bars
    roll_high_shift = np.roll(roll_high, 1)
    roll_low_shift = np.roll(roll_low, 1)
    roll_high_shift[0] = np.nan
    roll_low_shift[0] = np.nan
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, roll_high_shift)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, roll_low_shift)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period Donchian + 50-period EMA + 12h shift)
    start_idx = 20 + 50 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above Donchian high + bullish weekly trend + volume spike
        if close[i] > donch_high_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below Donchian low + bearish weekly trend + volume spike
        elif close[i] < donch_low_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Donchian level
        elif position == 1 and close[i] < donch_low_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donch_high_aligned[i]:
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

name = "12h_Donchian20_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0