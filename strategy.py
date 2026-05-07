#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_1dTrend_Volume
# Hypothesis: Uses weekly pivot points from 1w data to identify key support/resistance levels.
# Enters long when price breaks above weekly R1 with 1d uptrend (price > 1d EMA50) and volume spike.
# Enters short when price breaks below weekly S1 with 1d downtrend (price < 1d EMA50) and volume spike.
# Weekly pivots provide structure that works in both bull and bear markets; 1d trend filter ensures
# alignment with intermediate trend; volume confirms breakout strength. Targets 15-30 trades/year.

name = "6h_WeeklyPivot_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1w data for weekly pivot points (HTF as specified)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter on 6h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema_50_1d_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Price > weekly R1, above 1d EMA50 trend, volume spike
            if close[i] > r1_6h[i] and close[i] > ema_50_1d_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price < weekly S1, below 1d EMA50 trend, volume spike
            elif close[i] < s1_6h[i] and close[i] < ema_50_1d_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price breaks below weekly R1 or below 1d EMA50
            if close[i] < r1_6h[i] or close[i] < ema_50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above weekly S1 or above 1d EMA50
            if close[i] > s1_6h[i] or close[i] > ema_50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals