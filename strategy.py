#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike
Hypothesis: On 6h timeframe, enter long when price breaks above 20-period Donchian high AND weekly pivot is bullish (weekly close > weekly open) AND volume > 2.0x 20-period average volume. Enter short when price breaks below 20-period Donchian low AND weekly pivot is bearish (weekly close < weekly open) AND volume > 2.0x 20-period average volume. Exit on opposite Donchian breakout or volume dry-up. Uses discrete sizing (0.0, ±0.25) to limit fee drag. Target: 12-37 trades/year. Weekly pivot provides structural bias that works in both bull and bear markets.
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
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot direction: bullish if weekly close > weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish weekly candle
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Calculate 20-period Donchian channels on 6h data
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: fixed threshold of 2.0x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian and volume MA warmup
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(volume_ma[i])):
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
        
        # Weekly pivot direction
        weekly_dir_bullish = weekly_bullish_aligned[i] > 0.5
        weekly_dir_bearish = weekly_bullish_aligned[i] <= 0.5
        
        if position == 0:
            # Long: bullish breakout + weekly bullish pivot + volume spike
            long_signal = breakout_up and weekly_dir_bullish and volume_spike[i]
            
            # Short: bearish breakout + weekly bearish pivot + volume spike
            short_signal = breakout_down and weekly_dir_bearish and volume_spike[i]
            
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
            # Exit: bearish Donchian breakout OR volume dry-up (< 1.5x average)
            if breakout_down or volume[i] < 1.5 * volume_ma[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: bullish Donchian breakout OR volume dry-up (< 1.5x average)
            if breakout_up or volume[i] < 1.5 * volume_ma[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike"
timeframe = "6h"
leverage = 1.0