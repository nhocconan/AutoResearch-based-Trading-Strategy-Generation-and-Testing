#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm_v1
Hypothesis: Trade 6h Donchian(20) breakouts aligned with weekly pivot direction (from 1w HTF) with volume confirmation.
In bull markets (price above weekly pivot): buy when price breaks above 20-period Donchian high with volume > 1.5x 20-period average.
In bear markets (price below weekly pivot): sell when price breaks below 20-period Donchian low with volume > 1.5x 20-period average.
Weekly pivot acts as regime filter: only trade long above pivot, short below pivot.
Volume confirmation reduces false breakouts.
Position size: 0.25 to limit drawdown.
Target: 12-25 trades/year to stay well under 300-trade 6h hard max.
Works in bull (breakouts with uptrend regime) and bear (breakdowns with downtrend regime).
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
    
    # Get 1w data for weekly pivot calculation (HTF regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor pivot: P = (H+L+C)/3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 6h volume average (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().values
    volume_spike = volume > (1.5 * avg_volume)  # 50% above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and volume avg (20)
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine regime from weekly pivot
        price_above_pivot = close[i] > pivot_1w_aligned[i]
        price_below_pivot = close[i] < pivot_1w_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian high + above weekly pivot + volume spike
            long_setup = (close[i] > highest_high[i]) and price_above_pivot and volume_spike[i]
            
            # Short setup: price breaks below Donchian low + below weekly pivot + volume spike
            short_setup = (close[i] < lowest_low[i]) and price_below_pivot and volume_spike[i]
            
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
            # Exit: price touches Donchian low (stop) OR price goes below weekly pivot (regime change)
            if (close[i] <= lowest_low[i]) or (not price_above_pivot):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian high (stop) OR price goes above weekly pivot (regime change)
            if (close[i] >= highest_high[i]) or (price_above_pivot):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0