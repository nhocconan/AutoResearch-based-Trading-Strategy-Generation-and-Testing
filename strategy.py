#!/usr/bin/env python3
"""
6H_Donchian20_WeeklyTrend_VolumeFilter
Hypothesis: 6h Donchian(20) breakouts filtered by 1-week EMA50 trend and volume confirmation.
Works in both bull and bear: breakouts capture momentum, weekly EMA50 filters direction,
volume confirms breakout strength. Targets 50-150 total trades over 4 years to minimize fee drag.
"""
name = "6H_Donchian20_WeeklyTrend_VolumeFilter"
timeframe = "6h"
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
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1W EMA50 for trend direction
    close_1w_series = pd.Series(df_1w['close'])
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) on 6h: highest high and lowest low of last 20 bars
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current 6h volume > 1.8 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 48 bars between trades (12 days on 6h TF) to reduce frequency
            if bars_since_exit < 48:
                continue
                
            # Long: price breaks above Donchian high with weekly uptrend and volume spike
            if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and 
                close[i] > ema_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below Donchian low with weekly downtrend and volume spike
            elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and 
                  close[i] < ema_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Donchian side (mean reversion or trend end)
            if position == 1 and close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals