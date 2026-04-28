#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotTrend_Volume
Hypothesis: 6-hour Donchian(20) breakouts aligned with weekly pivot trend and volume confirmation. Targets 15-35 trades/year by requiring breakout in direction of weekly trend with volume surge. Works in bull (breakouts continue) and bear (breakouts reverse) markets.
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
    
    # Get weekly data for pivot trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot_point - prev_week_low
    s1 = 2 * pivot_point - prev_week_high
    
    # Weekly trend: price above/below pivot
    weekly_trend_up = prev_week_close > pivot_point
    weekly_trend_down = prev_week_close < pivot_point
    
    # Align weekly data to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    # Donchian(20) on 6h data
    lookback = 20
    dc_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    dc_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(weekly_trend_down_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: price breaks above Donchian high + weekly uptrend + volume surge
        long_entry = (high[i] > dc_high[i] and 
                     weekly_trend_up_aligned[i] > 0.5 and 
                     volume_surge[i])
        
        # Short: price breaks below Donchian low + weekly downtrend + volume surge
        short_entry = (low[i] < dc_low[i] and 
                      weekly_trend_down_aligned[i] > 0.5 and 
                      volume_surge[i])
        
        # Exit on opposite Donchian break with volume surge
        long_exit = low[i] < dc_low[i] and volume_surge[i]
        short_exit = high[i] > dc_high[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_WeeklyPivotTrend_Volume"
timeframe = "6h"
leverage = 1.0