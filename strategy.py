#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_VolumeConfirm_v1
Hypothesis: Trade 6h Donchian(20) breakouts aligned with weekly pivot direction and volume confirmation. 
In bullish weekly trend (price above weekly pivot), buy breakouts above upper Donchian; 
in bearish weekly trend (price below weekly pivot), sell breakdowns below lower Donchian. 
Volume confirmation (2.0x 50-bar avg) filters false breakouts. 
Designed for 6h timeframe with moderate frequency (~20-40 trades/year) to balance edge and fee drag.
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
    
    # Get 1d data for HTF indicators (weekly pivot from daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's OHLC
    # Approximate weekly OHLC from daily: 
    # Weekly high = max of last 5 daily highs
    # Weekly low = min of last 5 daily lows
    # Weekly close = last daily close
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling window of 5 days for weekly aggregation
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (prior week's pivot available)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Donchian channel (20-period) on 6h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 50-bar average volume
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20) and volume MA(50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly trend
        weekly_bullish = close[i] > weekly_pivot_aligned[i]
        weekly_bearish = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation
            long_breakout = (high[i] > donchian_upper[i]) and volume_spike[i]
            short_breakout = (low[i] < donchian_lower[i]) and volume_spike[i]
            
            # Only trade in direction of weekly trend
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
            # Exit when price retouches Donchian midpoint or weekly trend reverses
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2.0
            exit_signal = (low[i] < donchian_mid) or (not weekly_bullish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price retouches Donchian midpoint or weekly trend reverses
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2.0
            exit_signal = (high[i] > donchian_mid) or weekly_bullish
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0