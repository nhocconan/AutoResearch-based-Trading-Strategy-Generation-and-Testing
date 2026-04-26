#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeConfirm_v1
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (bullish/bearish bias) and volume confirmation. Weekly pivot provides structural bias from higher timeframe, Donchian captures breakouts, volume confirms validity. Works in bull markets (breakouts with bias) and bear markets (breakdowns with bias). Target: 50-150 total trades over 4 years (12-38/year).
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
    
    # Load weekly data ONCE before loop for pivot direction
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Standard pivot: P = (H + L + C)/3
    # Support 1: S1 = 2*P - H
    # Resistance 1: R1 = 2*P - L
    if len(df_1w) < 2:
        return np.zeros(n)
    
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly pivot calculation
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - prev_week_high
    weekly_s1 = 2 * weekly_pivot - prev_week_low
    
    # Determine weekly bias: bullish if close above pivot, bearish if below
    weekly_bullish = prev_week_close > weekly_pivot
    weekly_bearish = prev_week_close < weekly_pivot
    
    # Align weekly bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Donchian channel (20-period) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = volume > (volume_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Long logic: price breaks above Donchian high with volume + weekly bullish bias
        if high[i] > highest_high[i] and volume_confirm[i] and weekly_bullish_aligned[i] > 0.5:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below Donchian low with volume + weekly bearish bias
        elif low[i] < lowest_low[i] and volume_confirm[i] and weekly_bearish_aligned[i] > 0.5:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite Donchian level or weekly bias flips
        elif position == 1 and (low[i] < lowest_low[i] or weekly_bearish_aligned[i] > 0.5):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (high[i] > highest_high[i] or weekly_bullish_aligned[i] > 0.5):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0