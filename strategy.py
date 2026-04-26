#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation (>2.0x median) to capture strong directional moves while filtering noise. Enters long when price breaks above 20-day high with volume confirmation and bullish weekly trend. Enters short when price breaks below 20-day low with volume confirmation and bearish weekly trend. Exits on opposite Donchian breakout. Uses discrete position sizing (0.25) to minimize churn. Target: 30-100 trades over 4 years. Works in both bull and bear markets by following weekly trend filter.
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
    
    # Calculate Donchian(20) channels from daily data (based on previous daily bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous bar's values for channel calculation (to avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    
    # Calculate 20-period Donchian channels
    highest_high_20 = pd.Series(high_1d_prev).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d_prev).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d primary timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    
    # Volume confirmation: volume > 2.0x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    # Load weekly data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period Donchian, 50-period volume median, 50-period weekly EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above 20-day high + volume confirmation + bullish weekly trend
        if close[i] > highest_high_20_aligned[i] and volume_confirm[i] and close[i] > ema_50_1w_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below 20-day low + volume confirmation + bearish weekly trend
        elif close[i] < lowest_low_20_aligned[i] and volume_confirm[i] and close[i] < ema_50_1w_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite Donchian breakout
        elif position == 1 and close[i] < lowest_low_20_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high_20_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0