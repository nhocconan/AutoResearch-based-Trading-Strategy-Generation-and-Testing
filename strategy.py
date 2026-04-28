#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeSpike
Hypothesis: Combines Donchian(20) breakout on 6h with weekly pivot direction (from 1w data) and volume spike (2x 24-bar average) to capture high-probability breakouts. Weekly pivot direction ensures we only trade in the direction of the weekly trend, reducing false signals in sideways markets. Volume spike confirms institutional interest. Works in both bull and bear by following weekly trend direction. Targets 15-25 trades/year via strict conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C)/3
    # Support 1 = (2*Pivot) - High
    # Resistance 1 = (2*Pivot) - Low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = (2 * weekly_pivot) - weekly_high
    weekly_s1 = (2 * weekly_pivot) - weekly_low
    
    # Determine weekly bias: bullish if close above pivot, bearish if below
    weekly_bullish = weekly_close > weekly_pivot
    weekly_bearish = weekly_close < weekly_pivot
    
    # Align weekly data to 6h timeframe (with 1-week delay for completed bar)
    weekly_bullish_aligned = align_ltf_to_htf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_ltf_to_htf(prices, df_1w, weekly_bearish.astype(float))
    weekly_r1_aligned = align_ltf_to_htf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_ltf_to_htf(prices, df_1w, weekly_s1)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2x 24-period MA (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_24[i])
        
        # Weekly filter: only trade in direction of weekly bias
        weekly_long_filter = weekly_bullish_aligned[i] > 0.5
        weekly_short_filter = weekly_bearish_aligned[i] > 0.5
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and weekly_long_filter
        short_entry = short_breakout and vol_confirm and weekly_short_filter
        
        # Exit conditions: opposite Donchian breakout
        long_exit = short_breakout  # Exit long on downside breakout
        short_exit = long_breakout  # Exit short on upside breakout
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_VolumeSpike"
timeframe = "6h"
leverage = 1.0