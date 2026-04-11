#!/usr/bin/env python3
# 6h_12h_camarilla_pivot_volume_v1
# Strategy: 6h Camarilla pivot levels from 12h timeframe with volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) act as strong support/resistance.
# In ranging markets, price tends to revert from R3/S3 levels.
# In trending markets, breakouts beyond R4/S4 with volume confirmation continue the trend.
# Works in both bull and bear markets by adapting to regime via volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    typical_price = (high + low + close) / 3
    # Range
    range_val = high - low
    # Pivot point
    pivot = typical_price
    # Camarilla levels
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    r3_12h = np.zeros(len(df_12h))
    r4_12h = np.zeros(len(df_12h))
    s3_12h = np.zeros(len(df_12h))
    s4_12h = np.zeros(len(df_12h))
    
    for i in range(len(df_12h)):
        r3, r4, s3, s4 = calculate_camarilla(high_12h[i], low_12h[i], close_12h[i])
        r3_12h[i] = r3
        r4_12h[i] = r4
        s3_12h[i] = s3
        s4_12h[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price action relative to Camarilla levels
        price = close[i]
        r3 = r3_12h_aligned[i]
        r4 = r4_12h_aligned[i]
        s3 = s3_12h_aligned[i]
        s4 = s4_12h_aligned[i]
        
        # Entry logic
        # Long setup: price rejects S3/S4 with volume OR breaks above R4 with volume
        long_reject = (price > s3 and price < s3 * 1.005) and vol_confirm[i]  # Near S3 bounce
        long_breakout = price > r4 and vol_confirm[i]  # Break above R4
        
        # Short setup: price rejects R3/R4 with volume OR breaks below S4 with volume
        short_reject = (price < r3 and price > r3 * 0.995) and vol_confirm[i]  # Near R3 rejection
        short_breakout = price < s4 and vol_confirm[i]  # Break below S4
        
        if (long_reject or long_breakout) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (short_reject or short_breakout) and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to mid-range or opposite rejection
        elif position == 1 and (price < (r3 + s3) / 2 or short_reject):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (price > (r3 + s3) / 2 or long_reject):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals