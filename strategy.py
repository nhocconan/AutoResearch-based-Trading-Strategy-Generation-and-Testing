#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout in direction of weekly EMA50 trend with volume spike confirmation.
Enters long when price breaks above 20-day high with bullish weekly trend and volume spike.
Enters short when price breaks below 20-day low with bearish weekly trend and volume spike.
Uses ATR-based stoploss and discrete position sizing (0.0, ±0.30) to minimize fee churn.
Designed for 30-100 total trades over 4 years (7-25/year) on BTC/ETH/SOL.
Works in both bull and bear markets by following weekly trend direction only.
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First bar has no prior close
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Donchian(20) channels - using prior completed daily bar to avoid look-ahead
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Start after warmup (need Donchian20 + weekly EMA50)
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above 20-day high + bullish weekly trend + volume spike
        if close[i] > highest_20[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below 20-day low + bearish weekly trend + volume spike
        elif close[i] < lowest_20[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Stoploss: ATR-based exit
        elif position == 1 and close[i] <= highest_20[i] - 2.0 * atr[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] >= lowest_20[i] + 2.0 * atr[i]:
            signals[i] = 0.0
            position = 0
        # Exit: price reverts to opposite Donchian level
        elif position == 1 and close[i] < lowest_20[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_20[i]:
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

name = "1d_Donchian20_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0