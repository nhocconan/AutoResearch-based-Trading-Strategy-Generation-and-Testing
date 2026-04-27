#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_S4_S5_Range_Reversion
Mean reversion on Camarilla pivot S4/S5 levels with 1w trend filter.
Long when price touches S4 in 1w uptrend, short when touches S5 in 1w downtrend.
Exit at S3 or R3 respectively, or when trend fails.
Target: 10-25 trades/year per symbol.
"""

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
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 12)
    # S2 = C - (Range * 1.1 / 6)
    # S3 = C - (Range * 1.1 / 4)
    # S4 = C - (Range * 1.1 / 2)
    # S5 = C - (Range * 1.1)
    # R3 = C + (Range * 1.1 / 4)
    # R4 = C + (Range * 1.1 / 2)
    # R5 = C + (Range * 1.1)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    s5_1d = close_1d - (range_1d * 1.1)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    # Align 1d Camarilla levels to 1d timeframe (no shift needed as we use previous day's levels)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    s5_1d_aligned = align_htf_to_ltf(prices, df_1d, s5_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_1w_period:
        ema_1w[ema_1w_period - 1] = np.mean(close_1w[:ema_1w_period])
        for i in range(ema_1w_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_1w_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_1w_period + 1))))
    
    # Align 1w EMA50 to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need at least 1 day of data
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(s4_1d_aligned[i]) or np.isnan(s5_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        s4 = s4_1d_aligned[i]
        s5 = s5_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        ema1w = ema_1w_aligned[i]
        
        if position == 0:
            # Long: price touches or goes below S4 in 1w uptrend
            if price <= s4 and price > ema1w:
                signals[i] = size
                position = 1
            # Short: price touches or goes above S5 in 1w downtrend
            elif price >= s5 and price < ema1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches S3 or trend fails
            if price >= s3 or price <= ema1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches R3 or trend fails
            if price <= r3 or price >= ema1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_Pivot_S4_S5_Range_Reversion"
timeframe = "1d"
leverage = 1.0