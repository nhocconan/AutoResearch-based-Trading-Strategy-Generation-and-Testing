#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction (from 1w) and volume confirmation (>1.5x 20-bar mean volume). Long when price breaks above Donchian high and weekly pivot is bullish (price > weekly pivot); short when price breaks below Donchian low and weekly pivot is bearish (price < weekly pivot). Uses discrete position sizing (0.25) to minimize fee churn. Weekly pivot provides structural bias to reduce whipsaw in ranging markets while capturing breakouts with conviction. Designed for 15-25 trades/year per symbol, effective in both bull (breakouts with volume) and bear (trend-following via shorts) markets.
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
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Weekly pivot = (Prior week High + Prior week Low + Prior week Close) / 3
    weekly_pivot = (df_1w['high'].shift(1) + df_1w['low'].shift(1) + df_1w['close'].shift(1)) / 3
    weekly_pivot_vals = weekly_pivot.values
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_vals)
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > weekly pivot (bullish bias) with volume confirmation
            # Short: price breaks below Donchian low AND price < weekly pivot (bearish bias) with volume confirmation
            long_signal = (close[i] > donchian_high[i]) and (close[i] > weekly_pivot_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] < donchian_low[i]) and (close[i] < weekly_pivot_aligned[i]) and vol_confirm[i]
            
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
            # Exit when price moves back below Donchian mean (mean reversion exit)
            donchian_mean = (donchian_high[i] + donchian_low[i]) / 2
            exit_signal = close[i] < donchian_mean
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian mean (mean reversion exit)
            donchian_mean = (donchian_high[i] + donchian_low[i]) / 2
            exit_signal = close[i] > donchian_mean
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0