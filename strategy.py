#!/usr/bin/env python3
# 1d_Donchian20_Breakout_1wTrend_Volume
# Hypothesis: Daily Donchian(20) breakouts with weekly trend filter and volume confirmation capture strong directional moves.
# In bull markets: breakouts above upper band in weekly uptrend. In bear markets: breakouts below lower band in weekly downtrend.
# Volume filter ensures breakouts have institutional participation. Low trade frequency expected due to multi-condition confluence.
# Target: 20-50 trades per year on 1d timeframe.

name = "1d_Donchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to daily timeframe (already daily data)
    upper_20_aligned = upper_20  # no alignment needed for same timeframe
    lower_20_aligned = lower_20
    
    # Get daily data for volume filter
    volume = df_1d['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (50) + Donchian (20) + volume EMA (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_20_aligned[i]) or
            np.isnan(lower_20_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from weekly EMA
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        uptrend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian in weekly uptrend with volume
            if close_1d[i] > upper_20_aligned[i] and uptrend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian in weekly downtrend with volume
            elif close_1d[i] < lower_20_aligned[i] and downtrend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below lower Donchian or trend changes to downtrend
            if close_1d[i] < lower_20_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above upper Donchian or trend changes to uptrend
            if close_1d[i] > upper_20_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals