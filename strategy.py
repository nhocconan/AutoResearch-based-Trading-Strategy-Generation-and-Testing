#!/usr/bin/env python3
"""
12h Camarilla Pivot R3/S3 Breakout with Volume Spike and Daily Trend Filter.
Long when price breaks above R3 with volume spike and daily EMA34 uptrend.
Short when price breaks below S3 with volume spike and daily EMA34 downtrend.
Exit when price reverts to daily EMA34 or breaks opposite S3/R3 level.
Designed to generate 15-35 trades/year per symbol with strong institutional level-based edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(values, period):
    """Exponential Moving Average"""
    n = len(values)
    result = np.empty(n, dtype=np.float64)
    result.fill(np.nan)
    if n < period:
        return result
    alpha = 2.0 / (period + 1.0)
    result[0] = values[0]
    for i in range(1, n):
        result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's range
    # Using previous day's high, low, close (standard Camarilla formula)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R3 = pivot + (range_hl * 1.1 / 2.0)
    S3 = pivot - (range_hl * 1.1 / 2.0)
    R4 = pivot + (range_hl * 1.1)
    S4 = pivot - (range_hl * 1.1)
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily EMA34 for trend filter
    ema34 = calculate_ema(df_1d['close'].values, 34)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume filter: volume > 2.0x average (to capture institutional interest)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla calculation (needs previous day) + EMA34 + volume MA
    start_idx = max(34, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        ema34_val = ema34_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and daily uptrend
            if price_now > r3 and vol_filter and price_now > ema34_val:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with volume spike and daily downtrend
            elif price_now < s3 and vol_filter and price_now < ema34_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S3 (reversion) or falls below daily EMA34
            if price_now < s3 or price_now < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R3 (reversion) or rises above daily EMA34
            if price_now > r3 or price_now > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_Volume_EMA34"
timeframe = "12h"
leverage = 1.0