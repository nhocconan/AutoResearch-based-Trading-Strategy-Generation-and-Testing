#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Donchian channels identify breakouts from recent price ranges. Weekly pivot (from prior week)
# provides directional bias: long only above weekly pivot, short only below. Volume surge
# confirms institutional participation. This avoids whipsaws in ranging markets and works
# in both bull/bear by aligning with higher-timeframe structure. Targets 12-25 trades/year.

name = "6h_Donchian20_WeeklyPivot_Volume"
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
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader's pivot)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L, S1 = 2*PP - H
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pp = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pp - low_w
    s1 = 2 * pp - high_w
    
    # Align weekly pivot to 6h (use prior week's levels)
    pp_6h = align_htf_to_ltf(prices, df_1w, pp)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    
    # Donchian channel (20 periods) on 6h
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 24-period average (~4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 24)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(pp_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(vol_surge[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, above weekly pivot, volume surge
            if close[i] > highest[i] and close[i] > pp_6h[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, below weekly pivot, volume surge
            elif close[i] < lowest[i] and close[i] < pp_6h[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly pivot or Donchian low
            if close[i] < pp_6h[i] or close[i] < lowest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly pivot or Donchian high
            if close[i] > pp_6h[i] or close[i] > highest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals