#!/usr/bin/env python3
"""
4h_12h_pivots_v1
Hypothesis: Use 12-hour pivot points (standard, not Camarilla) with price position relative to pivot and R1/S1, combined with 12-hour EMA50 trend filter and volume confirmation. Works in bull/bear by requiring price to be on correct side of pivot relative to trend, avoiding counter-trend entries. Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""
name = "4h_12h_pivots_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and pivot points
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend direction
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Previous 12h bar's OHLC for standard pivot points
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_open_12h = np.roll(df_12h['open'].values, 1)
    
    # Standard pivot point: P = (H + L + C) / 3
    pivot = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    # Support and resistance levels
    r1 = 2 * pivot - prev_low_12h
    s1 = 2 * pivot - prev_high_12h
    r2 = pivot + (prev_high_12h - prev_low_12h)
    s2 = pivot - (prev_high_12h - prev_low_12h)
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price > EMA50 (uptrend) AND price > pivot AND close breaks above R1 with volume
        if (close[i] > ema50_12h_aligned[i] and close[i] > pivot_aligned[i] and close[i] > r1_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price < EMA50 (downtrend) AND price < pivot AND close breaks below S1 with volume
        elif (close[i] < ema50_12h_aligned[i] and close[i] < pivot_aligned[i] and close[i] < s1_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or price crosses back to opposite S1/R1
        elif position == 1 and close[i] < s1_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r1_aligned[i]:
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