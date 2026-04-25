#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm_v1
Hypothesis: Trade 6h Donchian(20) breakouts aligned with weekly pivot direction (from 1d HTF) and volume confirmation. 
In bullish weekly trend (price above weekly pivot), buy breakouts above 20-bar high; 
In bearish weekly trend (price below weekly pivot), sell breakdowns below 20-bar low. 
Volume spike (2.0x 20-bar avg) confirms institutional participation. 
Designed for 6h timeframe with tight entries (~15-25/year) to minimize fee drag while capturing strong directional moves.
Uses discrete position sizing (0.25) to reduce churn. Works in both bull and bear markets by following weekly pivot direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation (using prior week's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Need to resample 1d to weekly - but we can approximate using rolling window
    # Weekly high = max of prior 7 daily highs, weekly low = min of prior 7 daily lows, weekly close = prior 7th daily close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate rolling weekly OHLC (using 7-day window, shifted by 1 to use prior week only)
    weekly_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().shift(1).values
    weekly_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().shift(1).values
    weekly_close = pd.Series(close_1d).rolling(window=7, min_periods=7).last().shift(1).values
    
    # Weekly pivot point = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20), volume MA(20), and weekly data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly trend direction from pivot
        weekly_bullish = close[i] > weekly_pivot_aligned[i]
        weekly_bearish = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and weekly pivot alignment
            long_breakout = (high[i] > donchian_high[i]) and volume_spike[i]
            short_breakout = (low[i] < donchian_low[i]) and volume_spike[i]
            
            # Only trade in direction of weekly pivot trend
            if long_breakout and weekly_bullish:
                signals[i] = 0.25
                position = 1
            elif short_breakout and weekly_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price returns to weekly pivot or Donchian middle
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            exit_signal = (low[i] < donchian_mid) or (not weekly_bullish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price returns to weekly pivot or Donchian middle
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            exit_signal = (high[i] > donchian_mid) or weekly_bullish
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0