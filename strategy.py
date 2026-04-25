#!/usr/bin/env python3
"""
6h_WeeklyPivotDir_DailyDonchian_Breakout_VolumeConfirm
Hypothesis: 6h breakout of daily Donchian(20) channels with weekly pivot direction filter and volume confirmation (>2.0x 20-bar mean volume). Weekly pivot direction (price > weekly pivot = bullish bias for longs, < weekly pivot = bearish bias for shorts) provides HTF regime alignment. Daily Donchian breakout captures intermediate-term momentum. Volume confirmation ensures breakouts have conviction. Designed for 12-25 trades/year per symbol, effective in bull markets (breakouts with volume) and bear markets (shorts on breakdowns with volume). Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Get daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 6h timeframe (use previous day's levels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate weekly pivot point (using prior weekly bar)
    # Pivot = (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    weekly_pivot_vals = weekly_pivot.values
    
    # Align weekly pivot to 6h timeframe (use previous week's pivot)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_vals)
    
    # Volume confirmation: current volume > 2.0x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above daily Donchian high with weekly bullish bias (price > weekly pivot) and volume confirmation
            long_signal = (close[i] > donchian_high_aligned[i]) and (close[i] > weekly_pivot_aligned[i]) and vol_confirm[i]
            # Short: price breaks below daily Donchian low with weekly bearish bias (price < weekly pivot) and volume confirmation
            short_signal = (close[i] < donchian_low_aligned[i]) and (close[i] < weekly_pivot_aligned[i]) and vol_confirm[i]
            
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
            # Exit when price moves back below daily Donchian low (breakdown)
            exit_signal = close[i] < donchian_low_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above daily Donchian high (breakout)
            exit_signal = close[i] > donchian_high_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivotDir_DailyDonchian_Breakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0