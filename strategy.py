#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_camarilla_pivot
# Uses weekly Camarilla pivot levels from 1w and daily trend from 1d as filters.
# Long when price breaks above weekly R4 with daily close above 20-period EMA.
# Short when price breaks below weekly S4 with daily close below 20-period EMA.
# Exits when price crosses weekly pivot (central level).
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drift.
# Works in trending markets via breakouts and ranges via mean reversion to pivot.
# Focus on BTC/ETH as primary targets.

name = "6h_1w_1d_camarilla_pivot"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot levels using previous week's OHLC
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r4_1w = close_1w + (range_1w * 1.1 / 2)
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    r2_1w = close_1w + (range_1w * 1.1 / 6)
    r1_1w = close_1w + (range_1w * 1.1 / 12)
    s1_1w = close_1w - (range_1w * 1.1 / 12)
    s2_1w = close_1w - (range_1w * 1.1 / 6)
    s3_1w = close_1w - (range_1w * 1.1 / 4)
    s4_1w = close_1w - (range_1w * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Daily trend filter: 20-period EMA
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly R4 with daily uptrend
        if (close[i] > r4_1w_aligned[i] and close[i] > ema_20_1d_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly S4 with daily downtrend
        elif (close[i] < s4_1w_aligned[i] and close[i] < ema_20_1d_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses weekly pivot (mean reversion)
        elif position == 1 and close[i] <= pivot_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pivot_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals