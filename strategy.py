#!/usr/bin/env python3
"""
6h_WeeklyPivotDir_DailyDonchian_Breakout_VolumeConfirm
Hypothesis: Use 1w pivot direction (based on weekly close vs weekly open) as trend filter, and 1d Donchian(20) breakout with volume confirmation for entry on 6h timeframe. Weekly pivot direction provides robust trend bias that works in both bull and bear markets. 1d Donchian breakout captures multi-day momentum, and volume confirmation reduces false breakouts. Designed for low trade frequency (12-30/year) with discrete sizing to minimize fee drag.
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
    
    # Get 1w data for trend filter (weekly pivot direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot direction: 1 if weekly close > open (bullish), -1 if close < open (bearish)
    weekly_dir = np.where(close_1w > open_1w, 1, -1)
    
    # Align weekly direction to 6h timeframe
    weekly_dir_aligned = align_htf_to_ltf(prices, df_1w, weekly_dir)
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) on 1d: upper = max(high, 20), lower = min(low, 20)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: current volume > 1.5x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_dir_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with bullish weekly direction and volume confirmation
            # Short: price breaks below 1d Donchian low with bearish weekly direction and volume confirmation
            long_signal = (close[i] > donchian_high_aligned[i]) and (weekly_dir_aligned[i] == 1) and vol_confirm[i]
            short_signal = (close[i] < donchian_low_aligned[i]) and (weekly_dir_aligned[i] == -1) and vol_confirm[i]
            
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
            # Exit when price moves back below 1d Donchian low (mean reversion) or weekly direction turns bearish
            exit_signal = (close[i] < donchian_low_aligned[i]) or (weekly_dir_aligned[i] == -1)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d Donchian high (mean reversion) or weekly direction turns bullish
            exit_signal = (close[i] > donchian_high_aligned[i]) or (weekly_dir_aligned[i] == 1)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivotDir_DailyDonchian_Breakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0