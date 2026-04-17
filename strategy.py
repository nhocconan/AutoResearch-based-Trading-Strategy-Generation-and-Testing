#!/usr/bin/env python3
"""
12h_Donchian20_1dTrend_VolumeFilter
Strategy: 12h Donchian channel breakout with 1d trend filter and volume confirmation.
Long: Close > 20-period high + 1d uptrend + volume > 1.5x 20-period average
Short: Close < 20-period low + 1d downtrend + volume > 1.5x 20-period average
Exit: Close crosses back below/above 10-period moving average
Position size: 0.25
Designed to capture trends with volume confirmation and trend filter.
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h 10-period moving average for exit
    ma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # 12h 20-period volume average for confirmation
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d trend data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d trend: close > open = uptrend (1), close < open = downtrend (0)
    trend_1d = (df_1d['close'] > df_1d['open']).astype(float).values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(20, n):  # warmup for Donchian
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma10[i]) or np.isnan(volume_ma20[i]) or 
            np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter
        trend_up = trend_1d_aligned[i] > 0.5  # 1d uptrend
        trend_down = trend_1d_aligned[i] < 0.5  # 1d downtrend
        
        # Entry signals
        if position == 0:
            # Long: break above Donchian high + volume + uptrend
            if close[i] > donchian_high[i] and volume_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + volume + downtrend
            elif close[i] < donchian_low[i] and volume_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close crosses below 10-period MA
            if close[i] < ma10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close crosses above 10-period MA
            if close[i] > ma10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0