#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
Long when price breaks above Donchian(20) high and 1-day close > 1-day SMA(50).
Short when price breaks below Donchian(20) low and 1-day close < 1-day SMA(50).
Exit when price crosses Donchian middle band.
Uses volume confirmation: current volume > 1.5x 20-period average volume.
Designed to capture trends while avoiding chop, works in both bull and bear markets by following institutional-grade breakouts.
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
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.5x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(sma_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume + 1-day uptrend
            if (high[i] > highest_high[i] and 
                volume_filter[i] and 
                close[i] > sma_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume + 1-day downtrend
            elif (low[i] < lowest_low[i] and 
                  volume_filter[i] and 
                  close[i] < sma_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Donchian middle band
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below middle
                if close[i] < middle[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above middle
                if close[i] > middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0