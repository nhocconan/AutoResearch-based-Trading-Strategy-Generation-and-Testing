#!/usr/bin/env python3
# 6h_1d_1w_pivot_trend_v1
# Strategy: 6h trading using daily and weekly pivot levels with trend alignment
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Price reacts to daily and weekly pivot levels (support/resistance).
# In uptrend (price > weekly pivot), buy at daily S1/S2 with stop at S3.
# In downtrend (price < weekly pivot), sell at daily R1/R2 with stop at R3.
# Uses 1d and 1w pivots for institutional levels, reducing whipsaw.
# Targets 2-4 trades/month (24-48/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_pivot_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P, R1, R2, S1, S2, R3, S3"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 10 or len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate daily pivot points
    dh = df_1d['high'].values
    dl = df_1d['low'].values
    dc = df_1d['close'].values
    dp, dr1, dr2, dr3, ds1, ds2, ds3 = calculate_pivot_points(dh, dl, dc)
    
    # Calculate weekly pivot points
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    wp, wr1, wr2, wr3, ws1, ws2, ws3 = calculate_pivot_points(wh, wl, wc)
    
    # Align pivot levels to 6h timeframe
    p_1d = align_htf_to_ltf(prices, df_1d, dp)
    r1_1d = align_htf_to_ltf(prices, df_1d, dr1)
    r2_1d = align_htf_to_ltf(prices, df_1d, dr2)
    r3_1d = align_htf_to_ltf(prices, df_1d, dr3)
    s1_1d = align_htf_to_ltf(prices, df_1d, ds1)
    s2_1d = align_htf_to_ltf(prices, df_1d, ds2)
    s3_1d = align_htf_to_ltf(prices, df_1d, ds3)
    
    p_1w = align_htf_to_ltf(prices, df_1w, wp)
    r1_1w = align_htf_to_ltf(prices, df_1w, wr1)
    r2_1w = align_htf_to_ltf(prices, df_1w, wr2)
    r3_1w = align_htf_to_ltf(prices, df_1w, wr3)
    s1_1w = align_htf_to_ltf(prices, df_1w, ws1)
    s2_1w = align_htf_to_ltf(prices, df_1w, ws2)
    s3_1w = align_htf_to_ltf(prices, df_1w, ws3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(p_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or \
           np.isnan(p_1w[i]) or np.isnan(r1_1w[i]) or np.isnan(s1_1w[i]) or \
           np.isnan(vol_confirm[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend based on weekly pivot
        uptrend = close[i] > p_1w[i]
        downtrend = close[i] < p_1w[i]
        
        # Long conditions: uptrend + price at daily support + volume
        long_setup = (
            uptrend and 
            (close[i] <= s1_1d[i] * 1.005 or close[i] <= s2_1d[i] * 1.005) and  # Near S1 or S2
            vol_confirm[i]
        )
        
        # Short conditions: downtrend + price at daily resistance + volume
        short_setup = (
            downtrend and 
            (close[i] >= r1_1d[i] * 0.995 or close[i] >= r2_1d[i] * 0.995) and  # Near R1 or R2
            vol_confirm[i]
        )
        
        # Exit conditions: opposite pivot level or trend change
        exit_long = (
            position == 1 and 
            (close[i] >= r1_1d[i] or uptrend == False)  # Hit R1 or trend change
        )
        
        exit_short = (
            position == -1 and 
            (close[i] <= s1_1d[i] or downtrend == False)  # Hit S1 or trend change
        )
        
        # Entry logic
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit logic
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals