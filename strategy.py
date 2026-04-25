#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: Donchian channel breakouts capture strong momentum, while weekly pivot
direction provides higher-timeframe bias, and volume confirmation filters false breakouts.
This combination works in both bull and bear markets by trading with the weekly trend.
Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
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
    
    # Get weekly data for pivot points (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Weekly trend: price above/below pivot
    weekly_trend_above = weekly_close > weekly_pivot
    weekly_trend_below = weekly_close < weekly_pivot
    
    # Align weekly data to 6h timeframe (waits for weekly bar to close)
    weekly_trend_above_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_above.astype(float))
    weekly_trend_below_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_below.astype(float))
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate 20-period Donchian channels on 6h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for 6h volume spike
    vol_ma_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_6h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20_6h[i]) or
            np.isnan(weekly_trend_above_aligned[i]) or np.isnan(weekly_trend_below_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_ma_6h = vol_ma_20_6h[i]
        weekly_above = weekly_trend_above_aligned[i] > 0.5
        weekly_below = weekly_trend_below_aligned[i] > 0.5
        weekly_r1 = weekly_r1_aligned[i]
        weekly_s1 = weekly_s1_aligned[i]
        
        # Volume confirmation: current 6h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_6h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND weekly trend is up AND volume confirmation
            long_entry = (curr_close > donch_high and 
                         weekly_above and 
                         volume_confirm)
            # Short: price breaks below Donchian low AND weekly trend is down AND volume confirmation
            short_entry = (curr_close < donch_low and 
                          weekly_below and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian low OR weekly trend turns down
            if curr_close < donch_low or not weekly_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high OR weekly trend turns up
            if curr_close > donch_high or not weekly_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0