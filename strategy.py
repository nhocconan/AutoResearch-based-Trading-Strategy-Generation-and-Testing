#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm_v2
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot trend (price above/below weekly pivot) and volume confirmation (>1.5x 20-bar avg). Uses discrete sizing (0.25) to limit trades (~20-40/year) and avoid fee drag. Weekly pivot provides robust trend filter that works in both bull and bear markets. Volume confirmation ensures breakout momentum. Designed for BTC/ETH robustness with tight entry conditions.
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
    
    # Get 1d data for HTF trend filter and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot from prior week (using last 5 daily bars: Mon-Fri)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # We'll use rolling window of 5 days to approximate weekly
    if len(close_1d) >= 5:
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values  # prior week
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    else:
        weekly_pivot = np.full_like(close_1d, np.nan)
    
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate 20-bar average volume for confirmation on 6h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly pivot (5 days) and volume MA20
    start_idx = 20  # volume MA20 needs 20 bars, weekly pivot needs 5*24=120 6h bars but we align from daily
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-bar average
            volume_confirm = volume[i] > 1.5 * vol_ma20[i]
            
            # Calculate 6h Donchian channels (20-bar)
            donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().iloc[i]
            donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().iloc[i]
            
            # Long: price breaks above Donchian(20) high AND above weekly pivot with volume
            # Short: price breaks below Donchian(20) low AND below weekly pivot with volume
            long_signal = (close[i] > donchian_high) and (close[i] > weekly_pivot_aligned[i]) and volume_confirm
            short_signal = (close[i] < donchian_low) and (close[i] < weekly_pivot_aligned[i]) and volume_confirm
            
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
            # Exit when price moves back below weekly pivot (trend reversal)
            exit_signal = close[i] < weekly_pivot_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above weekly pivot (trend reversal)
            exit_signal = close[i] > weekly_pivot_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0