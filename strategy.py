#!/usr/bin/env python3
# 6h_12h_Donchian20_Breakout_WeeklyPivotDirection_Volume
# Hypothesis: 6h Donchian(20) breakout in direction of 1w pivot (R1/S1), with volume confirmation.
# Weekly pivot provides directional bias (bullish/bearish) to avoid counter-trend breakouts.
# Volume filter ensures breakouts have participation. Works in bull/bear by aligning with higher-timeframe bias.
# Target: 15-30 trades/year per symbol.

name = "6h_12h_Donchian20_Breakout_WeeklyPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for Donchian calculation (more responsive than 6h alone)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper/lower: highest high/lowest low of last 20 periods
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot points from daily data (using last 5 days)
    # Weekly pivot = (weekly high + weekly low + weekly close) / 3
    # We approximate using last 5 trading days
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close using 5-day rolling window
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values  # last close in window
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    weekly_r1 = weekly_pivot + (weekly_range * 1.1 / 12)  # R1
    weekly_s1 = weekly_pivot - (weekly_range * 1.1 / 12)  # S1
    
    # Calculate volume average for spike detection (24 periods = 4 days of 12h)
    vol_ma_12h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align all 12h indicators to 6h timeframe
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5 * 12h average volume
        volume_spike = volume[i] > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly R1 (bullish bias) with volume
            if close[i] > donch_high_12h_aligned[i] and close[i] > weekly_r1_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below weekly S1 (bearish bias) with volume
            elif close[i] < donch_low_12h_aligned[i] and close[i] < weekly_s1_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low (breakdown) or below weekly S1
            if close[i] < donch_low_12h_aligned[i] or close[i] < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high (breakout) or above weekly R1
            if close[i] > donch_high_12h_aligned[i] or close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals