#!/usr/bin/env python3
"""
6H_Donchian_Breakout_20_WeeklyPivotDirection_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakouts with weekly pivot point (from weekly high/low/close) direction filter and volume confirmation capture institutional breakout moves. Weekly pivot provides multi-day structure to filter false breakouts, while volume confirmation ensures follow-through. Designed for low trade frequency (15-35/year) to work in both bull (breakout continuation) and bear (mean reversion at extremes) markets by using pivot direction as regime filter.
"""

name = "6H_Donchian_Breakout_20_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Previous weekly bar for pivot calculation (use complete prior week)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_weekly, pivot_w)
    
    # Donchian channels (20-period) on 6m data
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback-1, 19)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot_w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high AND price > weekly pivot (bullish bias)
            if (high[i] > highest_high[i-1] and 
                close[i] > pivot_w_aligned[i] and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND price < weekly pivot (bearish bias)
            elif (low[i] < lowest_low[i-1] and 
                  close[i] < pivot_w_aligned[i] and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below Donchian low (reversal signal)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above Donchian high (reversal signal)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals