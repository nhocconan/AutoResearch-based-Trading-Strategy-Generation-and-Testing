#!/usr/bin/env python3
"""
4h_1d_1w_VolumeSpike_TrendBreak_v1
4-hour strategy combining 1-day trend direction with 1-week volume spike and price breakout of 4-hour range.
Goes long when: 1d close > 1d open (bullish), weekly volume > 1.5x 4-week average, and price breaks above 4h high of prior 20 bars.
Goes short when: 1d close < 1d open (bearish), weekly volume > 1.5x 4-week average, and price breaks below 4h low of prior 20 bars.
Uses volume spike as regime filter to avoid chop and ensure momentum.
Designed for low trade frequency (<50/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1-day Trend Direction ===
    df_1d = get_htf_data(prices, '1d')
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    # Bullish if close > open, bearish if close < open
    trend_bullish = close_1d > open_1d
    trend_bearish = close_1d < open_1d
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish)
    
    # === 1-week Volume Spike Filter ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    # 4-week average volume (using 4 prior weekly bars)
    vol_avg_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    vol_spike = volume_1w > 1.5 * vol_avg_1w
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike)
    
    # === 4-hour Price Channel (20-bar high/low) ===
    # Use prior 20 bars to avoid look-ahead
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or
            np.isnan(high_max_20[i]) or
            np.isnan(low_min_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions
        long_entry = trend_bullish_aligned[i] and vol_spike_aligned[i] and close[i] > high_max_20[i]
        short_entry = trend_bearish_aligned[i] and vol_spike_aligned[i] and close[i] < low_min_20[i]
        
        # Exit conditions: reverse signal or opposite breakout
        long_exit = trend_bearish_aligned[i] or close[i] < low_min_20[i]
        short_exit = trend_bullish_aligned[i] or close[i] > high_max_20[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                continue
            elif short_entry:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_1w_VolumeSpike_TrendBreak_v1"
timeframe = "4h"
leverage = 1.0