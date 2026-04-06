#!/usr/bin/env python3
"""
4h Donchian(20) breakout with 12h EMA200 trend filter and volume confirmation v1
Hypothesis: Donchian breakouts capture trend continuations; EMA200 on 12h filters
counter-trend trades (only long above EMA200, short below); volume confirms breakout
strength. Designed for 100-200 trades over 4 years to balance opportunity and cost.
Works in bull via breakouts, in bear via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_ema200_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA200 trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA200 on 12h
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema200_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian break
        if position == 1:  # long position
            # Exit: price closes below lower Donchian band
            if close[i] < lowest_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above upper Donchian band
            if close[i] > highest_high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + EMA200 trend filter + volume
            bull_break = close[i] > highest_high_20[i]
            bear_break = close[i] < lowest_low_20[i]
            vol_filter = volume[i] > vol_ma[i] * 1.5
            
            # Long: price breaks above upper band, above EMA200, with volume
            if bull_break and close[i] > ema200_12h_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, below EMA200, with volume
            elif bear_break and close[i] < ema200_12h_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals