#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
- Long: Price breaks above Donchian(20) high + price > 1d weekly pivot R1 + volume > 2x 20-period average
- Short: Price breaks below Donchian(20) low + price < 1d weekly pivot S1 + volume > 2x 20-period average
- Exit: Price crosses Donchian(20) midpoint (mean reversion to avoid whipsaws)
- Uses weekly pivot levels from 1d timeframe for institutional reference points
- Volume confirmation ensures institutional participation
- Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
- Works in both bull and bear markets by trading breakouts with institutional level filters
"""

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
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Get 1d data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week
    # Weekly pivot: (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # R1 = (2 * Pivot) - Prior Week Low
    # S1 = (2 * Pivot) - Prior Week High
    # We need to shift by 1 week to avoid look-ahead
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values  # Prior week high
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values      # Prior week low
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values  # Prior week close
    
    # Weekly pivot calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = (2 * pivot) - weekly_low  # Resistance 1
    s1 = (2 * pivot) - weekly_high  # Support 1
    
    # Align HTF pivot levels to LTF
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 5)  # Donchian needs 20, weekly pivot needs 5 (for prior week calc)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above prior period high
        breakout_down = close[i] < lowest_low[i-1]   # Break below prior period low
        
        # Weekly pivot filter
        above_r1 = close[i] > r1_aligned[i]
        below_s1 = close[i] < s1_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + above weekly R1 + volume spike
            if breakout_up and above_r1 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + below weekly S1 + volume spike
            elif breakout_down and below_s1 and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Donchian midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian midpoint
                if close[i] < donchian_mid[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price rises above Donchian midpoint
                if close[i] > donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_1dWeeklyPivot_R1S1_Breakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0