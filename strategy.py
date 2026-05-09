#!/usr/bin/env python3
# 1d_WeeklyDonchian20_Breakout_WeeklyTrend_Filter
# Hypothesis: On daily timeframe, breakout of weekly Donchian(20) channels with weekly trend filter.
# Long when weekly trend up (price > weekly SMA50) and price breaks above weekly Donchian upper.
# Short when weekly trend down (price < weekly SMA50) and price breaks below weekly Donchian lower.
# Uses weekly trend to avoid counter-trend trades and reduce whipsaw in ranging markets.
# Weekly timeframe reduces trade frequency to avoid fee drag; daily execution allows timely entry.

name = "1d_WeeklyDonchian20_Breakout_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate weekly SMA50 for trend filter
    sma50_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 50:
        sma50_weekly[49] = np.mean(close_weekly[0:50])
        for i in range(50, len(close_weekly)):
            sma50_weekly[i] = (sma50_weekly[i-1] * 49 + close_weekly[i]) / 50
    
    # Calculate weekly Donchian(20) channels
    donchian_high = np.full_like(high_weekly, np.nan)
    donchian_low = np.full_like(low_weekly, np.nan)
    
    for i in range(len(close_weekly)):
        if i >= 19:
            donchian_high[i] = np.max(high_weekly[i-19:i+1])
            donchian_low[i] = np.min(low_weekly[i-19:i+1])
    
    # Align weekly indicators to daily timeframe
    sma50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, sma50_weekly)
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need weekly SMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma50_weekly_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close[i] > sma50_weekly_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + price breaks above weekly Donchian high
            if trend_up and close[i] > donchian_high_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + price breaks below weekly Donchian low
            elif not trend_up and close[i] < donchian_low_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price breaks below weekly Donchian low
            if not trend_up or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price breaks above weekly Donchian high
            if trend_up or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals