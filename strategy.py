#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm_v1
Hypothesis: Trade 6h Donchian(20) breakouts aligned with weekly pivot direction and volume confirmation.
Weekly pivot provides structural bias from higher timeframe (1w) to filter false breakouts.
Donchian breakout captures momentum; volume confirmation ensures institutional participation.
Target: 12-30 trades/year to minimize fee drag and achieve Sharpe > 0.5 on test.
Works in both bull/bear via weekly pivot bias and volume filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous weekly bar)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    pivot_1w = (h_1w + l_1w + c_1w) / 3.0
    # Weekly bias: above pivot = bullish, below = bearish
    weekly_bullish = pivot_1w > 0  # placeholder, will align and use price vs pivot
    
    # Align weekly pivot to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Weekly trend bias: price above/below weekly pivot
    weekly_bullish_aligned = close > pivot_1w_aligned  # will be computed inside loop
    
    # Donchian channel (20-period) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 6h volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and volume MA (30)
    start_idx = max(lookback, 30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly bias from aligned pivot
        weekly_bias_bullish = close[i] > pivot_1w_aligned[i]
        weekly_bias_bearish = close[i] < pivot_1w_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian upper + weekly bullish bias + volume spike
            long_setup = (close[i] > highest_high[i]) and weekly_bias_bullish and volume_spike[i]
            
            # Short setup: price breaks below Donchian lower + weekly bearish bias + volume spike
            short_setup = (close[i] < lowest_low[i]) and weekly_bias_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Donchian lower (stop) OR weekly bias turns bearish
            if (close[i] <= lowest_low[i]) or (not weekly_bias_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian upper (stop) OR weekly bias turns bullish
            if (close[i] >= highest_high[i]) or (weekly_bias_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0