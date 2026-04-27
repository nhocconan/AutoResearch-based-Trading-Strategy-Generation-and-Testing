#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Filter_Volume
Hypothesis: Combine 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
Weekly pivot provides long-term trend filter; Donchian provides entry timing; volume confirms momentum.
Target: 15-25 trades/year per symbol. Works in bull (breakouts) and bear (fade false breaks via weekly filter).
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
    
    # Get weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly typical price and range for pivot calculation
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    range_ = df_1w['high'] - df_1w['low']
    
    # Weekly pivot point and support/resistance levels
    pp = typical_price
    r1 = pp + (range_ * 1.0 / 3)
    s1 = pp - (range_ * 1.0 / 3)
    r2 = pp + range_
    s2 = pp - range_
    
    # Align weekly levels to 6h timeframe (use previous week's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    
    # Weekly trend: price above/below pivot
    weekly_bull = pp_aligned > 0  # placeholder, will be replaced with actual comparison
    
    # 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian and weekly alignment
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend based on price vs pivot
        weekly_bull = close[i] > pp_aligned
        weekly_bear = close[i] < pp_aligned
        
        if position == 0:
            # Long: break above Donchian high + weekly bull + volume spike
            if high[i] > high_20[i] and weekly_bull and vol_spike[i]:
                signals[i] = size
                position = 1
            # Short: break below Donchian low + weekly bear + volume spike
            elif low[i] < low_20[i] and weekly_bear and vol_spike[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Donchian low or weekly turns bear
            if close[i] < low_20[i] or not weekly_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above Donchian high or weekly turns bull
            if close[i] > high_20[i] or not weekly_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Filter_Volume"
timeframe = "6h"
leverage = 1.0