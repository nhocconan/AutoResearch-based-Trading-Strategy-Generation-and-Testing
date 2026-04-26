#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Only long when price breaks above Donchian(20) high and 1w EMA50 rising, short when price breaks below Donchian(20) low and 1w EMA50 falling.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).
Designed to capture medium-term trends while filtering noise with volume and multi-timeframe trend confirmation.
Works in both bull and bear markets by combining price structure (Donchian) with higher timeframe trend (1w EMA) and volume filters.
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
    
    # Calculate Donchian(20) channels on primary timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate EMA50 slope for trend confirmation (rising/falling)
    ema_50_slope = np.diff(ema_50_1w_aligned, prepend=ema_50_1w_aligned[0])
    ema_rising = ema_50_slope > 0
    ema_falling = ema_50_slope < 0
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: price breaks above Donchian high + 1w EMA50 rising + volume spike
        if close[i] > highest_high[i] and ema_rising[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below Donchian low + 1w EMA50 falling + volume spike
        elif close[i] < lowest_low[i] and ema_falling[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price retracement to midpoint or loss of volume/Trend
        elif position == 1 and (close[i] < (highest_high[i] + lowest_low[i]) / 2 or not volume_spike[i] or not ema_rising[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > (highest_high[i] + lowest_low[i]) / 2 or not volume_spike[i] or not ema_falling[i]):
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

name = "12h_Donchian20_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0