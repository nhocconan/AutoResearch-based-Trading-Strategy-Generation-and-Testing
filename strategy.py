#!/usr/bin/env python3
"""
6h_weekly_pivot_reversion_v1
Hypothesis: Weekly pivot points (R1/S1) act as strong support/resistance on 6h timeframe.
In ranging markets, price reverts to weekly pivot with high probability.
In trending markets, price respects weekly pivot as dynamic support/resistance.
Volume confirmation filters false signals. Target: 15-35 trades/year (60-140 over 4 years).
Works in both bull/bear via adaptive entry: long when price > weekly pivot + above EMA200,
short when price < weekly pivot + below EMA200.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points and EMA200
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly OHLC for pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and support/resistance levels
    # Pivot = (H + L + C) / 3
    # R1 = (2 * Pivot) - L
    # S1 = (2 * Pivot) - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = (2 * pivot_1w) - low_1w
    s1_1w = (2 * pivot_1w) - high_1w
    
    # Weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    
    # Align weekly levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema200_6h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 20-period volume average on 6h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or np.isnan(ema200_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly pivot OR EMA200 turns down
            if close[i] < pivot_6h[i] or close[i] < ema200_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above weekly pivot OR EMA200 turns up
            if close[i] > pivot_6h[i] or close[i] > ema200_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long when: price > weekly pivot AND price > EMA200 (bullish bias)
            if (close[i] > pivot_6h[i] and 
                close[i] > ema200_6h[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short when: price < weekly pivot AND price < EMA200 (bearish bias)
            elif (close[i] < pivot_6h[i] and 
                  close[i] < ema200_6h[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals