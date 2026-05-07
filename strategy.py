#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_AMA_Volume
Hypothesis: Use daily EMA50 for trend direction and 12h Donchian channel (20) for breakout entries.
Long when price breaks above 12h upper Donchian and trend is up (price > daily EMA50).
Short when price breaks below 12h lower Donchian and trend is down (price < daily EMA50).
Volume confirmation: current volume > 1.3x 20-period average volume.
This combines trend-following with volatility-based breakouts to capture strong moves while avoiding chop.
Designed for 12h timeframe to limit trades (target 12-37/year) and work in both bull and bear markets.
"""
name = "12h_Donchian20_Breakout_1dTrend_AMA_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Get 12h data for Donchian channel (20)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    
    # Volume filter: current volume > 1.3x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian and trend is up
            if (high[i] > upper_aligned[i] and close[i] > ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian and trend is down
            elif (low[i] < lower_aligned[i] and close[i] < ema_50_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Donchian level
            if position == 1 and low[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and high[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals