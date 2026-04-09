#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_volume_v5
# Hypothesis: 6h strategy using weekly pivot points (from 1w) for structure, combined with 6h Donchian channel breakout and volume confirmation.
# Long: Price breaks above weekly R1 pivot AND above 6h Donchian upper channel (20), with volume > 1.5x 20-period average.
# Short: Price breaks below weekly S1 pivot AND below 6h Donchian lower channel (20), with volume > 1.5x 20-period average.
# Exit: Price crosses weekly main pivot (PP) or Donchian middle line.
# Uses weekly pivots for major support/resistance, Donchian for breakout confirmation, volume to filter false breakouts.
# Designed to work in both bull (breakouts) and bear (breakdowns) markets with tight entries to minimize fee drag.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_volume_v5"
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
    
    # Donchian channel (20-period) for 6h
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for weekly pivot points (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    
    # Align HTF weekly pivot points to 6h timeframe (wait for completed 1w bar)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below weekly PP OR below Donchian middle
            if close[i] < pp_aligned[i] or close[i] < donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above weekly PP OR above Donchian middle
            if close[i] > pp_aligned[i] or close[i] > donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above weekly R1 AND above Donchian upper, volume confirmed
            if (high[i] > r1_aligned[i] and close[i] > donchian_upper[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly S1 AND below Donchian lower, volume confirmed
            elif (low[i] < s1_aligned[i] and close[i] < donchian_lower[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals