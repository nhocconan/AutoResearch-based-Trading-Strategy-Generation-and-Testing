#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm
Hypothesis: 6h Donchian(20) breakouts in the direction of weekly pivot trend with volume confirmation.
Weekly pivot provides higher timeframe trend bias (bullish/bearish/neutral) to filter breakouts.
Volume spike (>2x 20-bar average) confirms breakout strength. Exits on reversion to 6h midline.
Discrete position sizing (0.25) minimizes fee churn. Target: 12-30 trades/year (50-120 total over 4 years).
Works in both bull/bear markets by aligning with weekly pivot direction.
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
    
    # Get 1d data for weekly pivot calculation (need 5 days for weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need at least 10 days for reasonable weekly pivot
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily OHLC (using prior week's data)
    # Weekly pivot = (PriorWeek High + PriorWeek Low + PriorWeek Close) / 3
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1)
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1)
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Trend direction: price above weekly pivot = bullish, below = bearish
    weekly_trend_bullish = weekly_pivot > 0  # Valid pivot exists
    weekly_trend_bias = np.where(weekly_close > weekly_pivot, 1,  # Bullish bias
                                np.where(weekly_close < weekly_pivot, -1, 0))  # Bearish or neutral
    
    # Get 6h data for Donchian channels and midline
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 25:  # Need 20 for Donchian + buffer
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Donchian(20) channels on 6h
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2  # Midline for exit
    
    # Align HTF indicators to 6h timeframe
    weekly_trend_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_trend_bias, additional_delay_bars=1)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high, additional_delay_bars=1)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low, additional_delay_bars=1)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_trend_bias_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        bias = int(weekly_trend_bias_aligned[i])  # -1, 0, or 1
        
        if position == 0:
            # Look for breakout signals in direction of weekly trend bias
            # Long: price breaks above Donchian HIGH with bullish bias + volume
            # Short: price breaks below Donchian LOW with bearish bias + volume
            long_signal = (bias == 1) and (close[i] > donchian_high_aligned[i]) and volume_spike[i]
            short_signal = (bias == -1) and (close[i] < donchian_low_aligned[i]) and volume_spike[i]
            
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
            # Exit when price moves back below Donchian midline (mean reversion)
            exit_signal = close[i] < donchian_mid_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian midline (mean reversion)
            exit_signal = close[i] > donchian_mid_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm"
timeframe = "6h"
leverage = 1.0