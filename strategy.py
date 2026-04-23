#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
Donchian channels identify volatility breakouts. Weekly pivot provides institutional bias.
Volume confirmation filters false breakouts. 6h timeframe reduces noise while capturing multi-day moves.
Works in bull (breakouts with trend) and bear (failed breakouts reverse quickly).
Target: 12-37 trades/year (50-150 over 4 years) with discrete sizing 0.25.
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
    
    # Calculate Donchian(20) from 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate weekly pivot from 1w data (requires 1w HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot: (prev_week_high + prev_week_low + prev_week_close) / 3
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    # Handle first bar
    prev_week_high[0] = df_1w['high'].iloc[0]
    prev_week_low[0] = df_1w['low'].iloc[0]
    prev_week_close[0] = df_1w['close'].iloc[0]
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)  # need Donchian, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Donchian upper AND price > weekly pivot (bullish bias) AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian lower AND price < weekly pivot (bearish bias) AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside Donchian channel OR loss of pivot bias
            exit_signal = False
            if position == 1:
                # Exit long when close < Donchian lower OR price < weekly pivot
                if close[i] < lowest_low[i] or close[i] < weekly_pivot_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > Donchian upper OR price > weekly pivot
                if close[i] > highest_high[i] or close[i] > weekly_pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_WeeklyPivot_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0