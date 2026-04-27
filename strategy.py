#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with 1d Trend Filter and Volume Spike
Long when price breaks above weekly R4 with 1d EMA50 uptrend and volume spike.
Short when price breaks below weekly S4 with 1d EMA50 downtrend and volume spike.
Exit when price crosses back below weekly pivot (long) or above weekly pivot (short).
Designed to capture strong momentum moves with low trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_pivots(high, low, close):
    """Calculate weekly pivot points (standard formula)"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def ema(arr, period):
    """Exponential Moving Average"""
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    wk_high = df_w['high'].values
    wk_low = df_w['low'].values
    wk_close = df_w['close'].values
    
    pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivots(wk_high, wk_low, wk_close)
    
    # Align weekly pivots to 6h
    pivot_6h = align_ltf_to_htf(prices, df_w, pivot)
    r4_6h = align_ltf_to_htf(prices, df_w, r4)
    s4_6h = align_ltf_to_htf(prices, df_w, s4)
    
    # Get daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_6h = align_ltf_to_htf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2x 20-period average
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly data + EMA50 + volume MA
    start_idx = max(19, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema50_6h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        pivot_val = pivot_6h[i]
        r4_val = r4_6h[i]
        s4_val = s4_6h[i]
        ema50_val = ema50_6h[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above weekly R4 + EMA50 uptrend + volume spike
            if price_now > r4_val and close[i-1] <= r4_val and ema50_val > close[i-20] and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly S4 + EMA50 downtrend + volume spike
            elif price_now < s4_val and close[i-1] >= s4_val and ema50_val < close[i-20] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below weekly pivot
            if price_now < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above weekly pivot
            if price_now > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_Pivot_R4S4_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0