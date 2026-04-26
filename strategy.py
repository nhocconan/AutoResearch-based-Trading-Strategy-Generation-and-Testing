#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (from 1w high/low) and volume confirmation. Weekly pivot provides structural bias (bull/bear) while Donchian captures breakouts. Volume filters false signals. Works in bull/bear via weekly trend filter. Target 12-30 trades/year (50-120 over 4 years). Discrete size 0.25 limits fee drag.
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
    
    # Get weekly data for pivot direction (using 1w high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot direction: based on previous week's range
    # Bullish bias: close above midpoint of weekly range
    # Bearish bias: close below midpoint of weekly range
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_midpoint = (weekly_high + weekly_low) / 2.0
    weekly_bullish = weekly_close > weekly_midpoint  # bullish bias for week
    weekly_bearish = weekly_close < weekly_midpoint   # bearish bias for week
    
    # Align weekly bias to 6h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.8x 30-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian(20), volume MA(30)
    start_idx = max(lookback, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > highest_high[i-1]  # break above previous Donchian high
        bearish_breakout = close[i] < lowest_low[i-1]   # break below previous Donchian low
        
        if position == 0:
            # Long: bullish breakout + weekly bullish bias + volume spike
            long_signal = bullish_breakout and weekly_bullish_aligned[i] > 0.5 and volume_spike[i]
            
            # Short: bearish breakout + weekly bearish bias + volume spike
            short_signal = bearish_breakout and weekly_bearish_aligned[i] > 0.5 and volume_spike[i]
            
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
            # Exit: bearish breakout (reverse signal) OR price touches Donchian low
            if bearish_breakout or close[i] <= lowest_low[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: bullish breakout (reverse signal) OR price touches Donchian high
            if bullish_breakout or close[i] >= highest_high[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0