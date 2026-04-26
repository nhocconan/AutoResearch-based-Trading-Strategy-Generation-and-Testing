#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakouts in direction of weekly pivot trend (price > weekly pivot = long bias, price < weekly pivot = short bias) with volume confirmation (>2.0x 20-period average). Weekly pivot acts as HTF regime filter to avoid counter-trend trades. Target: 12-25 trades/year. Designed to work in bull (breakouts with trend) and bear (fades at extremes) by aligning with weekly structure.
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
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for weekly pivot
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily OHLC (using prior week's data)
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    # We approximate by using rolling window on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close using 5-day window (prior complete week)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values  # shift(1) for prior week
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 6h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), volume MA (20), weekly pivot (5-day weekly)
    start_idx = max(20, 20)  # 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(weekly_pivot_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly pivot trend filter
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_pivot = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + price > weekly pivot + volume spike
            long_breakout = (close[i] > donchian_high[i]) and \
                           (close[i-1] <= donchian_high[i-1])  # Fresh breakout
            long_signal = long_breakout and price_above_pivot and volume_spike[i]
            
            # Short: price breaks below Donchian low + price < weekly pivot + volume spike
            short_breakout = (close[i] < donchian_low[i]) and \
                           (close[i-1] >= donchian_low[i-1])  # Fresh breakout
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
            # Exit: price touches Donchian low OR price < weekly pivot (trend change)
            if (close[i] < donchian_low[i] or not price_above_pivot):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Donchian high OR price > weekly pivot (trend change)
            if (close[i] > donchian_high[i] or not price_below_pivot):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0