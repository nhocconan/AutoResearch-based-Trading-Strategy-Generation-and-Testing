#!/usr/bin/env python3
"""
6H Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: 6H Donchian breakouts capture medium-term trends. Weekly pivot direction (from 1W high/low) filters for alignment with weekly bias, while volume confirmation ensures breakout strength. Designed for 50-150 trades over 4 years to minimize fee drag and work in both bull/bear markets via pivot-based trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot direction (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot: use weekly high and low as trend filter
    # If price > weekly high -> bullish bias, if price < weekly low -> bearish bias
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly trend: 1 if above weekly high, -1 if below weekly low, 0 otherwise
    weekly_trend = np.zeros(len(weekly_close))
    for i in range(len(weekly_close)):
        if weekly_close[i] > weekly_high[i]:
            weekly_trend[i] = 1
        elif weekly_close[i] < weekly_low[i]:
            weekly_trend[i] = -1
        else:
            weekly_trend[i] = 0
    
    # Align weekly trend to 6h timeframe
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter: 20-period moving average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or weekly trend reversal
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR weekly trend turns bearish
            if close[i] < donchian_low[i] or weekly_trend_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR weekly trend turns bullish
            if close[i] > donchian_high[i] or weekly_trend_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly trend alignment
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            
            # Only go long if weekly trend is bullish or neutral
            # Only go short if weekly trend is bearish or neutral
            weekly_bullish = weekly_trend_aligned[i] >= 0  # 0 or 1
            weekly_bearish = weekly_trend_aligned[i] <= 0  # 0 or -1
            
            if bull_breakout and volume_filter and weekly_bullish:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter and weekly_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals