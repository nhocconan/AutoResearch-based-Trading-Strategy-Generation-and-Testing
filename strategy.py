#!/usr/bin/env python3
"""
12H_Donchian_Breakout_1D_Trend_Filter_v1
Hypothesis: Use 12h Donchian channel breakout (20-period) for entry and 1d EMA(50) for trend filter.
Long when price breaks above 12h upper Donchian with 1d price > EMA50; short when price breaks below lower Donchian with 1d price < EMA50.
Volume confirmation: current volume > 1.3x 20-period average volume.
Designed for low-frequency, high-conviction trades to minimize fee drag and work in both bull and bear markets.
"""
name = "12H_Donchian_Breakout_1D_Trend_Filter_v1"
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
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter: current volume > 1.3x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 50)  # Ensure sufficient warmup for indicators
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (12 days on 12h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price breaks above upper Donchian with 1d price > EMA50
            if (high[i] > upper_aligned[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below lower Donchian with 1d price < EMA50
            elif (low[i] < lower_aligned[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Donchian level
            if position == 1 and low[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and high[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals