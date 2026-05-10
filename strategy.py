#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend_Volume
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation captures strong momentum moves while minimizing trades. Weekly trend ensures alignment with higher timeframe momentum, reducing false signals in ranging markets. Target: 15-25 trades/year per symbol for low fee drag and robust performance in both bull and bear markets.
"""

name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA40 for trend filter
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Calculate daily Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper (20-period high) and lower (20-period low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup periods
    start_idx = max(20, 20)  # Donchian(20) and volume EMA(20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema40_1w_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend_1w = close[i] > ema40_1w_aligned[i]
        downtrend_1w = close[i] < ema40_1w_aligned[i]
        
        # Volume filter
        volume_filter = volume[i] > vol_ema20[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above Donchian high with weekly uptrend and volume
            if high[i] > donchian_high[i] and uptrend_1w and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with weekly downtrend and volume
            elif low[i] < donchian_low[i] and downtrend_1w and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price retouches Donchian low or weekly trend fails
            if low[i] <= donchian_low[i] or not uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price retouches Donchian high or weekly trend fails
            if high[i] >= donchian_high[i] or not downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals