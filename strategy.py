#!/usr/bin/env python3
# 6H_WeeklyPivot_Trend_Filter_Volume
# Hypothesis: 6h strategy using weekly pivot points (R1/S1) with weekly trend filter and volume confirmation.
# Enters long when price breaks above weekly R1, close above weekly EMA34 (uptrend), and volume > 2x average.
# Enters short when price breaks below weekly S1, close below weekly EMA34 (downtrend), and volume > 2x average.
# Exits when price returns to opposite pivot level (S1 for longs, R1 for shorts).
# Weekly pivot points provide robust support/resistance levels that work in both bull and bear markets.
# Weekly trend filter ensures we only trade in the direction of higher timeframe trend, reducing whipsaw.
# Volume confirmation ensures we only trade on significant moves, reducing false breakouts.
# Target: 6h timeframe with weekly HTF for trend and levels, aiming for 12-37 trades per year.

name = "6H_WeeklyPivot_Trend_Filter_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot point calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point and support/resistance levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2.0 * pivot_1w - low_1w
    s1_1w = 2.0 * pivot_1w - high_1w
    
    # Align all levels to 6h timeframe (use previous week's levels)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate EMA34 for trend filter (weekly)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike detection: 2.0x average volume (24-period for stability on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # Ensure we have volume MA and EMA34 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1, close above weekly EMA34 (uptrend), volume spike (>2x)
            if (close[i] > r1_1w_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1, close below weekly EMA34 (downtrend), volume spike (>2x)
            elif (close[i] < s1_1w_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below weekly S1 (opposite level)
            if close[i] <= s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above weekly R1 (opposite level)
            if close[i] >= r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals