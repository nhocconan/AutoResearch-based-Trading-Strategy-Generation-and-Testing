#!/usr/bin/env python3
name = "6H_WeeklyPivot_Fibonacci_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Fibonacci pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly high, low, close for pivot
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Fibonacci extension levels (key breakout/retracement levels)
    # Resistance levels (for longs)
    r1_1w = pivot_1w + 0.382 * range_1w
    r2_1w = pivot_1w + 0.618 * range_1w
    r3_1w = pivot_1w + 1.000 * range_1w
    
    # Support levels (for shorts)
    s1_1w = pivot_1w - 0.382 * range_1w
    s2_1w = pivot_1w - 0.618 * range_1w
    s3_1w = pivot_1w - 1.000 * range_1w
    
    # Align weekly levels to 6h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 24-period average (4 days)
    volume_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R2 + above daily EMA50 + volume confirmation
            if close[i] > r2_1w_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S2 + below daily EMA50 + volume confirmation
            elif close[i] < s2_1w_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below R1 (failed breakout) OR below daily EMA50 (trend change)
            if close[i] < r1_1w_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above S1 (failed breakdown) OR above daily EMA50 (trend change)
            if close[i] > s1_1w_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals