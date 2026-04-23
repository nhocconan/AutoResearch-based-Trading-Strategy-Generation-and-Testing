#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
Donchian breakout captures momentum, weekly pivot (from 1d data) provides institutional bias,
volume confirmation filters false breakouts. 6h timeframe targets 12-37 trades/year.
Works in bull (breakouts with trend) and bear (fades at weekly pivot levels).
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
    
    # Calculate Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate weekly pivot from 1d data (using prior week's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Weekly high/low/close from prior 5 trading days (approximation)
    week_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (week_high + week_low + week_close) / 3.0
    
    # Align HTF data to 6h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), lowest_low)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 5, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian high AND price above weekly pivot (bullish bias) AND volume spike
            if (close[i] > highest_high_aligned[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume[i] > 1.3 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low AND price below weekly pivot (bearish bias) AND volume spike
            elif (close[i] < lowest_low_aligned[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume[i] > 1.3 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses weekly pivot (mean reversion to pivot) OR Donchian middle
            exit_signal = False
            if position == 1:
                # Exit long when price crosses below weekly pivot
                if close[i] < weekly_pivot_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price crosses above weekly pivot
                if close[i] > weekly_pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_WeeklyPivot_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0