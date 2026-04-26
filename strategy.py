#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirmation
Hypothesis: On 6h timeframe, enter long when price breaks above weekly Donchian(20) high AND weekly pivot shows bullish bias (close > weekly pivot) AND volume > 1.8x 20-period average. Enter short when price breaks below weekly Donchian(20) low AND weekly pivot shows bearish bias (close < weekly pivot) AND volume spike. Uses weekly structure for major trend alignment and Donchian breakouts for precise entries with volume confirmation. Designed for low trade frequency (12-25/year) with strong edge in both bull and bear markets via weekly trend filter and volatility-based exit.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and pivot calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian high: rolling max of high over 20 weekly periods
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low over 20 weekly periods
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point: (high + low + close) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly indicators to 6h timeframe (use previous weekly bar's values)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian warmup (20), volume MA warmup (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price above Donchian high + close > weekly pivot + volume spike
            long_signal = (close[i] > donchian_high_aligned[i] and 
                          close[i] > weekly_pivot_aligned[i] and 
                          volume_spike[i])
            
            # Short: price below Donchian low + close < weekly pivot + volume spike
            short_signal = (close[i] < donchian_low_aligned[i] and 
                           close[i] < weekly_pivot_aligned[i] and 
                           volume_spike[i])
            
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
            # Exit: price breaks below weekly pivot OR Donchian low breaks
            if close[i] < weekly_pivot_aligned[i] or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above weekly pivot OR Donchian high breaks
            if close[i] > weekly_pivot_aligned[i] or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0