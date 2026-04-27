#!/usr/bin/env python3
"""
#100827 - 6h_Donchian20_Breakout_1wPivotDirection_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Uses weekly pivot points to establish longer-term bias (bullish/bearish) and only takes breakouts
in the direction of the weekly trend. Volume confirmation ensures breakout strength.
Designed to work in both bull (breakouts with trend) and bear (mean reversion to pivot) markets
by filtering trades to the dominant weekly trend while avoiding counter-trend whipsaws.
Target: 15-30 trades/year to minimize fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data to avoid look-ahead)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    weekly_range = high_1w - low_1w
    weekly_r1 = weekly_pivot + weekly_range
    weekly_s1 = weekly_pivot - weekly_range
    
    # Align weekly pivot to 6h timeframe (previous week's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Determine weekly bias: bullish if close above pivot, bearish if below
    weekly_bias = np.where(close_1w > weekly_pivot, 1, -1)
    bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Donchian channels (20-period) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(bias_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian high, weekly bias bullish, volume spike
        if (high[i] > highest_high[i] and 
            bias_aligned[i] > 0 and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below Donchian low, weekly bias bearish, volume spike
        elif (low[i] < lowest_low[i] and 
              bias_aligned[i] < 0 and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to weekly pivot (mean reversion to longer-term average)
        elif position == 1 and close[i] < pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_1wPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0