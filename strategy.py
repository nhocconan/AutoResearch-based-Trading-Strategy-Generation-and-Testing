#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter (price above/below weekly pivot) and volume confirmation (>2.0x 20-bar avg). 
Enters long when price breaks above upper Donchian(20) in weekly uptrend (price > weekly pivot) with volume spike, short when breaks below lower Donchian(20) in weekly downtrend (price < weekly pivot) with volume spike. 
Exits on opposite Donchian touch (lower for longs, upper for shorts) or weekly trend reversal. 
Designed for 6h timeframe with ~12-37 trades/year, works in bull/bear by following weekly pivot trend filter.
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
    
    # Weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate Donchian channels for 6h timeframe using previous 20 bars' OHLC
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    lookback = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=lookback, min_periods=lookback).max().values
    lower = low_series.rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least lookback bars of previous data and weekly pivot warmup
    start_idx = max(lookback, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian in weekly uptrend (price > pivot) with volume confirmation
            long_setup = (close[i] > upper[i]) and (close[i] > pivot_1w_aligned[i]) and volume_spike[i]
            # Short: price breaks below lower Donchian in weekly downtrend (price < pivot) with volume confirmation
            short_setup = (close[i] < lower[i]) and (close[i] < pivot_1w_aligned[i]) and volume_spike[i]
            
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
            # Exit: price touches lower Donchian OR weekly trend turns down
            if (close[i] <= lower[i]) or (close[i] < pivot_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper Donchian OR weekly trend turns up
            if (close[i] >= upper[i]) or (close[i] > pivot_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0