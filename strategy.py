#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm_v1
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot trend direction and volume confirmation. Weekly pivot (calculated from prior week's HLC) provides robust trend filter that works in both bull and bear markets by identifying institutional support/resistance levels. Donchian breakouts capture momentum, while volume confirmation (>1.5x 20-bar average) ensures breakout strength. Designed for low trade frequency (~20-30/year) to minimize fee drag on 6h timeframe, with discrete sizing (0.25) to balance risk and return. The weekly pivot trend filter reduces whipsaws by only allowing breakouts in the direction of the weekly trend.
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
    
    # Get 1w data for weekly pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot points from prior week's OHLC
    # P = (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    # Weekly trend: price above/below weekly pivot
    weekly_trend = close_1w > weekly_pivot  # True for uptrend
    
    # Align weekly trend to 6h timeframe (use prior week's trend)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend.astype(float))
    
    # Calculate Donchian channels (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-bar average volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma20[i]) or
            np.isnan(weekly_trend_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-bar average
            volume_confirm = volume[i] > 1.5 * vol_ma20[i]
            
            # Long: price breaks above Donchian high in uptrend with volume confirmation
            # Short: price breaks below Donchian low in downtrend with volume confirmation
            long_signal = (close[i] > donchian_high[i]) and weekly_trend_aligned[i] and volume_confirm
            short_signal = (close[i] < donchian_low[i]) and (not weekly_trend_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Donchian low (breakdown)
            exit_signal = close[i] < donchian_low[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian high (breakout)
            exit_signal = close[i] > donchian_high[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0