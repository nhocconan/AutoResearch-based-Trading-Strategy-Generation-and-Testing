#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm
Hypothesis: Trade 6h Donchian(20) breakouts in direction of weekly pivot (from 1d HTF) with volume confirmation. Weekly pivot acts as regime filter: price above weekly pivot = bullish bias (long breakouts), price below = bearish bias (short breakouts). Volume spike confirms breakout strength. Designed for 6h timeframe to capture medium-term moves in both bull and bear markets. Target: 12-30 trades/year (50-120 over 4 years). Discrete size 0.25 limits fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot from prior week (using 1d data)
    # Weekly high = max(high of last 7 days), weekly low = min(low of last 7 days), weekly close = close of last day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling window of 7 days for weekly high/low
    weekly_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().shift(1).values  # prior week
    weekly_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().shift(1).values   # prior week
    weekly_close = pd.Series(close_1d).shift(1).values  # prior day's close as weekly close proxy
    
    # Weekly pivot point: (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Donchian(20) on 6h timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values  # prior 20 bars
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values   # prior 20 bars
    
    # Volume confirmation: volume > 2.0x 20-period average on 6h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # Align weekly pivot to 6h timeframe (wait for prior week to complete)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), volume MA (20), weekly pivot aligned
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: price relative to weekly pivot
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_pivot = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + price above weekly pivot + volume spike
            long_breakout = close[i] > donchian_high[i]
            long_signal = long_breakout and price_above_pivot and volume_spike[i]
            
            # Short: price breaks below Donchian low + price below weekly pivot + volume spike
            short_breakout = close[i] < donchian_low[i]
            short_signal = short_breakout and price_below_pivot and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches Donchian low OR weekly pivot turns bearish (price below pivot)
            if (close[i] < donchian_low[i] or not price_above_pivot):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Donchian high OR weekly pivot turns bullish (price above pivot)
            if (close[i] > donchian_high[i] or not price_below_pivot):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0