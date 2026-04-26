#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Weekly pivot (calculated from prior week OHLC) provides structural bias: long only when price
above weekly pivot, short only when below. Volume confirmation (>1.5x 20-period median) ensures
breakouts have conviction. Designed to work in both bull (breakouts continue) and bear
(breakdowns continue) markets by following the weekly pivot trend. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) for breakout signals
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (vol_median * 1.5)
    
    # Load weekly data for pivot calculation (prior week OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period for Donchian and volume median)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(weekly_pivot_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above Donchian high with volume confirmation and price > weekly pivot
        long_condition = (close[i] > highest_high[i]) and volume_confirm[i] and (close[i] > weekly_pivot_aligned[i])
        # Short logic: break below Donchian low with volume confirmation and price < weekly pivot
        short_condition = (close[i] < lowest_low[i]) and volume_confirm[i] and (close[i] < weekly_pivot_aligned[i])
        
        # Exit logic: opposite Donchian level touch or pivot cross
        exit_long = (close[i] < lowest_low[i]) or (close[i] < weekly_pivot_aligned[i])
        exit_short = (close[i] > highest_high[i]) or (close[i] > weekly_pivot_aligned[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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

name = "6h_Donchian20_Breakout_WeeklyPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0