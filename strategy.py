#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm_v1
Hypothesis: Trade 6h Donchian(20) breakouts in the direction of the weekly pivot trend with volume confirmation.
Weekly pivot trend: price above weekly pivot = bullish bias (long breakouts), below = bearish bias (short breakouts).
Volume confirmation: 6h volume > 1.8 * 20-period average volume to avoid false breakouts.
Position size: 0.25 (25% of capital) to balance profit and fee drag.
Target: 12-25 trades/year (~50-100 over 4 years) to stay well under 6h hard max of 300 total trades.
Works in bull markets via breakout continuation and in bear markets via breakdown continuation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous weekly bar)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    pivot_1w = (h_1w + l_1w + c_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe (use previous weekly bar's pivot)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 6h volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and volume MA (20)
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly pivot trend
        weekly_bullish = close[i] > pivot_1w_aligned[i]
        weekly_bearish = close[i] < pivot_1w_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian high + weekly bullish + volume spike
            long_setup = (close[i] > highest_high[i]) and weekly_bullish and volume_spike[i]
            
            # Short setup: price breaks below Donchian low + weekly bearish + volume spike
            short_setup = (close[i] < lowest_low[i]) and weekly_bearish and volume_spike[i]
            
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
            # Exit: price touches Donchian low (stop) OR weekly trend turns bearish
            if (close[i] <= lowest_low[i]) or (not weekly_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian high (stop) OR weekly trend turns bullish
            if (close[i] >= highest_high[i]) or (weekly_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0