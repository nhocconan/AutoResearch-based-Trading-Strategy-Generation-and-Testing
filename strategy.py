#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_breakout_volume_v1
# Hypothesis: 6h strategy using weekly pivot levels and Donchian(20) breakouts with volume confirmation.
# In both bull and bear markets, price tends to respect weekly pivot levels as support/resistance.
# Donchian breakouts provide momentum confirmation, while volume filters false breakouts.
# Weekly HTF ensures signals are aligned with major market structure, reducing whipsaw.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Weekly pivot support/resistance levels (standard pivot formulas)
    r1 = 2 * pivot_1w - low_1w
    s1 = 2 * pivot_1w - high_1w
    r2 = pivot_1w + range_1w
    s2 = pivot_1w - range_1w
    r3 = high_1w + 2 * (pivot_1w - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Donchian channel (20-period) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves below S1 or Donchian break down
            if close[i] < s1_aligned[i] or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above R1 or Donchian break up
            if close[i] > r1_aligned[i] or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above R1 with volume confirmation
                if close[i] > r1_aligned[i] and high[i] > r1_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below S1 with volume confirmation
                elif close[i] < s1_aligned[i] and low[i] < s1_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals