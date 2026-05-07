#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivot_Direction_Volume
Hypothesis: On 6h timeframe, use Donchian(20) breakouts filtered by weekly pivot direction (above/below weekly pivot) and volume confirmation. Weekly pivot provides directional bias from higher timeframe, reducing false breakouts in ranging markets. Volume confirmation ensures breakouts have institutional participation. Designed for 15-35 trades/year to minimize fee drag while capturing strong trends in both bull and bear markets.
"""
name = "6h_Donchian_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (based on prior week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot point (based on prior week)
    # Standard pivot: P = (H+L+C)/3
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    wp = (wh + wl + wc) / 3.0  # Weekly pivot
    
    # Align weekly pivot to 6h timeframe
    wp_aligned = align_htf_to_ltf(prices, df_1w, wp)
    
    # Get weekly trend: price above/below weekly pivot
    # Using prior week's pivot to avoid look-ahead
    weekly_bullish = wp_aligned > 0  # Will be replaced with actual comparison
    
    # Donchian channel (20-period) on 6h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume filter: current volume > 1.8 * 30-period average volume
    vol_avg = np.full(n, np.nan)
    for i in range(29, n):  # 30-period minimum
        vol_avg[i] = np.mean(volume[i - 29:i + 1])
    volume_filter = volume > (vol_avg * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(30, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(wp_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        # Calculate weekly bias: 1 if close > weekly pivot, -1 if close < weekly pivot
        weekly_bias = 1 if close[i] > wp_aligned[i] else -1
        
        if position == 0:
            # Minimum 12 bars between trades (3 days on 6h TF) to reduce frequency
            if bars_since_exit < 12:
                continue
                
            # Long: price breaks above Donchian high + weekly bullish bias + volume filter
            if (close[i] > highest_high[i] and 
                weekly_bias > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below Donchian low + weekly bearish bias + volume filter
            elif (close[i] < lowest_low[i] and 
                  weekly_bias < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Donchian level (mean reversion within channel)
            if position == 1 and close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals