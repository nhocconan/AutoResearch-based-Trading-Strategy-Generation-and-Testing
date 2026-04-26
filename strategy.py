#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation
Hypothesis: On 6h timeframe, enter long when price breaks above 20-period Donchian high AND weekly pivot bias is bullish (price > weekly pivot) AND volume > 1.5x 20-period average volume. Enter short when price breaks below 20-period Donchian low AND weekly pivot bias is bearish (price < weekly pivot) AND volume spike. Exit on opposite Donchian breakout or volume drying up. This combines price structure (Donchian), higher timeframe bias (weekly pivot), and volume confirmation to capture strong trending moves while filtering false breakouts. Weekly pivot provides structural bias that works in both bull and bear markets by identifying key support/resistance levels from higher timeframe.
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
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian warmup and volume MA warmup
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Weekly pivot bias
        pivot_bullish = close[i] > weekly_pivot_aligned[i]
        pivot_bearish = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout up + weekly pivot bullish + volume spike
            long_signal = breakout_up and pivot_bullish and volume_spike[i]
            
            # Short: Donchian breakout down + weekly pivot bearish + volume spike
            short_signal = breakout_down and pivot_bearish and volume_spike[i]
            
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
            # Exit: Donchian breakout down OR volume drying up (< average)
            if breakout_down or volume_spike[i] == False:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Donchian breakout up OR volume drying up
            if breakout_up or volume_spike[i] == False:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0