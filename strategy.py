#!/usr/bin/env python3
# 6h_donchian_breakout_weekly_pivot_volume_v1
# Hypothesis: 6h strategy combining Donchian(20) breakouts with weekly pivot direction and volume confirmation.
# In ranging markets (2025+), price tends to respect weekly pivot levels as support/resistance.
# Donchian breakouts provide directional momentum, filtered by weekly pivot bias and volume spikes.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for weekly pivot
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week (Mon-Fri)
    # For simplicity, use prior 5 trading days as proxy for week
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close from last 5 daily bars
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (week_high + week_low + week_close) / 3.0
    weekly_range = week_high - week_low
    
    # Weekly support/resistance levels (basic pivot)
    weekly_r1 = 2 * weekly_pivot - week_low
    weekly_s1 = 2 * weekly_pivot - week_high
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # Donchian channel (20-period) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves below weekly S1 or Donchian lower band
            if close[i] < s1_aligned[i] or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above weekly R1 or Donchian upper band
            if close[i] > r1_aligned[i] or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: Donchian breakout above upper band with weekly pivot bias (price > pivot)
                if close[i] > highest_high[i] and close[i] > pivot_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Donchian breakdown below lower band with weekly pivot bias (price < pivot)
                elif close[i] < lowest_low[i] and close[i] < pivot_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals