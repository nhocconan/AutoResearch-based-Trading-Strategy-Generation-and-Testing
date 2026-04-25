#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h Weekly Pivot Direction + Volume Spike
Hypothesis: Donchian breakouts capture momentum, while 12h weekly pivot direction (based on prior week's close) filters for institutional bias. Volume spike confirms participation. Works in bull/bear via pivot direction (long bias when price above weekly pivot, short bias when below). Targets 12-37 trades/year on 6h to minimize fee drag.
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
    
    # Get 12h data for weekly pivot calculation (using prior week's close)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:  # Need at least ~10 bars for weekly lookback
        return np.zeros(n)
    
    # Calculate weekly pivot points on 12h: PP = (Prior Week High + Low + Close) / 3
    # We'll use prior 5-day week approximation (5 * 12h = 60h ~ 2.5 days, adjust)
    # Simpler: use prior week's close as bias (if close > prior week close -> bullish bias)
    week_close = df_12h['close'].values
    # Shift by 1 to get prior week's close (each 12h bar = half day, so 14 bars ~ 1 week)
    prior_week_close = np.roll(week_close, 14)  # Approximate 1 week shift (14 * 12h = 168h = 1 week)
    prior_week_close[:14] = np.nan  # First 14 bars invalid
    
    # Pivot direction: 1 if current close > prior week close (bullish bias), -1 if < (bearish bias)
    pivot_bias = np.where(week_close > prior_week_close, 1, -1)
    pivot_bias = np.where(np.isnan(prior_week_close), 0, pivot_bias)
    pivot_bias_aligned = align_htf_to_ltf(prices, df_12h, pivot_bias)
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        window_high = high[i-lookback+1:i+1]
        window_low = low[i-lookback+1:i+1]
        highest_high[i] = np.max(window_high)
        lowest_low[i] = np.min(window_low)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    # For first 19 bars, use cumulative mean
    for i in range(19):
        vol_ma_20[i] = np.mean(volume[:i+1])
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after Donchian warmup
    start_idx = lookback - 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(pivot_bias_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        bias = pivot_bias_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND bullish bias AND volume spike
            long_condition = (curr_close > donchian_high) and (bias > 0) and vol_spike
            # Short: price breaks below Donchian low AND bearish bias AND volume spike
            short_condition = (curr_close < donchian_low) and (bias < 0) and vol_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below Donchian low or bias flips bearish
            if curr_close <= donchian_low or bias < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high or bias flips bullish
            if curr_close >= donchian_high or bias > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hWeeklyPivotDirection_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0