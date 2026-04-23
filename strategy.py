#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Donchian levels: Upper = 20-period high, Lower = 20-period low (on 6h timeframe)
- Long: price breaks above Donchian Upper + price > 1d EMA50 (uptrend) + volume > 2.0x 24-period avg
- Short: price breaks below Donchian Lower + price < 1d EMA50 (downtrend) + volume > 2.0x 24-period avg
- Exit: price crosses 1d EMA50 (trend-based exit)
- Uses 6h timeframe to capture medium-term breaks with lower frequency than 4h
- Volume confirmation ensures breakout validity and reduces false signals
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in both bull (breakouts with trend) and bear (filtered shorts in downtrends)
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
    
    # Volume confirmation: > 2.0x 24-period average (strict spike filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Donchian channels on 6h timeframe (primary)
    lookback = 20
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 24, 50)  # Need 20 for Donchian, 24 for volume MA, 50 for 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian Upper + price > 1d EMA50 (uptrend) + volume spike
            if volume_spike and close[i] > donchian_upper[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian Lower + price < 1d EMA50 (downtrend) + volume spike
            elif volume_spike and close[i] < donchian_lower[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1d EMA50 (trend-based exit)
            if close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1d EMA50 (trend-based exit)
            if close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0