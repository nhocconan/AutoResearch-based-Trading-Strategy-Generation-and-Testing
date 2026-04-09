#!/usr/bin/env python3
# 6h_donchian_weekly_pivot_volume_v1
# Hypothesis: 6h strategy using weekly pivot levels for trend direction and Donchian breakouts for entry.
# Uses volume confirmation to filter false breakouts. Works in bull/bear by using weekly pivots as structural levels.
# Weekly pivot provides higher timeframe bias, Donchian(20) captures breakouts, volume spike confirms momentum.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_weekly_pivot_volume_v1"
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
    
    # Get 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly support/resistance levels
    r1 = 2 * pivot_1w - low_1w
    s1 = 2 * pivot_1w - high_1w
    r2 = pivot_1w + range_1w
    s2 = pivot_1w - range_1w
    
    # Align weekly levels to 6h timeframe (completed 1w candle only)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Donchian channel (20-period) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike detection (20-period volume average on 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly S1 or Donchian lower band
            if close[i] < s1_aligned[i] or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above weekly R1 or Donchian upper band
            if close[i] > r1_aligned[i] or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian upper band with volume spike and above weekly pivot
            if (close[i] > highest_high[i]) and vol_spike[i] and (close[i] > pivot_1w[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower band with volume spike and below weekly pivot
            elif (close[i] < lowest_low[i]) and vol_spike[i] and (close[i] < pivot_1w[i]):
                position = -1
                signals[i] = -0.25
    
    return signals