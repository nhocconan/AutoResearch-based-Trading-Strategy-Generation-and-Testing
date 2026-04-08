#!/usr/bin/env python3
"""
12h Camarilla Pivot Strategy with Volume and 1d Trend Filter
Hypothesis: 12-hour strategy using daily Camarilla pivot levels (R3/S3) with volume confirmation 
and daily EMA trend filter. Enters long when price breaks above daily R3 with volume > 1.5x average 
and price > daily EMA200 (uptrend). Enters short when price breaks below daily S3 with volume > 1.5x 
average and price < daily EMA200 (downtrend). Exits when price crosses opposite daily level (S3 for long, R3 for short).
Uses higher timeframe pivots for better signal quality in both bull and bear markets.
Target: 15-30 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_v1"
timeframe = "12h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    ema = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily pivot (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # Daily support/resistance levels (Camarilla S3/R3)
    S3_1d = pivot_1d - (range_1d * 1.1 / 4)  # Daily S3
    R3_1d = pivot_1d + (range_1d * 1.1 / 4)  # Daily R3
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = calculate_ema(df_1d['close'].values, 200)
    
    # Align indicators to 12-hour timeframe
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: 50-period average
    vol_ma = np.full(n, np.nan)
    for i in range(50, n):
        vol_ma[i] = np.mean(volume[i-50:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(S3_1d_aligned[i]) or np.isnan(R3_1d_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        S3 = S3_1d_aligned[i]
        R3 = R3_1d_aligned[i]
        trend_up = price > ema_200_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below daily S3
            if price < S3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above daily R3
            if price > R3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above daily R3 with volume expansion and uptrend
            if price > R3 and vol_ratio > 1.5 and trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below daily S3 with volume expansion and downtrend
            elif price < S3 and vol_ratio > 1.5 and not trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals