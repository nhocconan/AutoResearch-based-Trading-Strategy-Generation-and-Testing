#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivotDirection_Volume
Hypothesis: Donchian channel breakouts on 6h timeframe with weekly pivot direction filter and volume confirmation capture institutional momentum. Weekly pivot acts as structural support/resistance, filtering false breakouts. Works in both bull and bear markets by aligning with higher timeframe bias. Target: 15-30 trades/year (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data
    # Weekly high/low/close (using last 5 days for weekly aggregation)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Weekly resistance 1: (2 * P) - L
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    # Weekly support 1: (2 * P) - H
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for Donchian and volume
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(weekly_pivot_6h[i]) or
            np.isnan(weekly_r1_6h[i]) or np.isnan(weekly_s1_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and price above weekly pivot
            if price > high_20[i] and vol_ok and price > weekly_pivot_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and price below weekly pivot
            elif price < low_20[i] and vol_ok and price < weekly_pivot_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly support or Donchian low
            if price < weekly_s1_6h[i] or price < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly resistance or Donchian high
            if price > weekly_r1_6h[i] or price > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0