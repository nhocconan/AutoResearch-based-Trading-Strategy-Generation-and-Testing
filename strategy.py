#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Donchian20_Breakout_TrendFilter
Hypothesis: On 6h timeframe, break above/below Donchian(20) channel with weekly pivot direction filter (1w trend) and volume confirmation (1.5x). 
Weekly pivot provides structural bias; Donchian breakout captures momentum; volume confirms institutional participation. 
Works in bull/bear via 1w trend alignment. Target: 12-37 trades/year (50-150 over 4 years). Size: 0.25.
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
    
    # Load 1w data ONCE before loop for weekly pivot and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week OHLC)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_open = df_1w['open'].shift(1).values
    
    pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Weekly R1/S1 (primary pivot levels)
    weekly_r1 = 2 * pivot_point - prev_week_low
    weekly_s1 = 2 * pivot_point - prev_week_high
    
    # Weekly trend: 1 if close > pivot, -1 if close < pivot
    weekly_trend_raw = np.where(prev_week_close > pivot_point, 1, -1)
    weekly_trend = align_htf_to_ltf(prices, df_1w, weekly_trend_raw)
    
    # Align weekly pivot levels
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Donchian channel (20-period) on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * volume_ma(30)
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 30 for volume MA)
    start_idx = max(20, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_trend[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high AND weekly uptrend AND volume spike
            if close[i] > highest_20[i] and weekly_trend[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND weekly downtrend AND volume spike
            elif close[i] < lowest_20[i] and weekly_trend[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR weekly trend turns down
            if close[i] < lowest_20[i] or weekly_trend[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR weekly trend turns up
            if close[i] > highest_20[i] or weekly_trend[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Weekly_Pivot_Donchian20_Breakout_TrendFilter"
timeframe = "6h"
leverage = 1.0