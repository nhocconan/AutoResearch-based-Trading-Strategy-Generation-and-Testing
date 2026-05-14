#!/usr/bin/env python3
"""
6H_CAMARILLA_R3_S3_BREAKOUT_1DVOLUMESPIKE_WKPIVOT
Hypothesis: Combine daily Camarilla R3/S3 breakout with weekly pivot direction filter to reduce whipsaw.
In bull markets (price above weekly pivot), only take longs from R3 breakout.
In bear markets (price below weekly pivot), only take shorts from S3 breakdown.
Volume spike (2.0x 20-period) confirms institutional participation.
Minimum 6-bar hold time prevents premature exits.
Target: 20-30 trades/year (80-120 total over 4 years) to stay within 6h limits.
Works in bull markets (breakouts continue with trend) and bear markets (mean reversion from extremes).
"""
name = "6H_CAMARILLA_R3_S3_BREAKOUT_1DVOLUMESPIKE_WKPIVOT"
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
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 2
    S3 = prev_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Weekly data for pivot direction filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot_vals = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_vals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(20, n):  # Start after warmup for volume MA
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # LONG: Price above weekly pivot (bullish bias) + break above R3 + volume spike
            if (close[i] > weekly_pivot_aligned[i] and 
                close[i] > R3_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # SHORT: Price below weekly pivot (bearish bias) + break below S3 + volume spike
            elif (close[i] < weekly_pivot_aligned[i] and 
                  close[i] < S3_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Only after minimum 6 bars AND (price re-enters range OR closes below weekly pivot)
            if bars_since_entry >= 6 and ((close[i] < R3_aligned[i] and close[i] > S3_aligned[i]) or 
                                          close[i] < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Only after minimum 6 bars AND (price re-enters range OR closes above weekly pivot)
            if bars_since_entry >= 6 and ((close[i] < R3_aligned[i] and close[i] > S3_aligned[i]) or 
                                          close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals