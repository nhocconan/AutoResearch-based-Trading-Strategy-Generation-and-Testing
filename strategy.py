#!/usr/bin/env python3
"""
6h_1d_camarilla_pivot_v1
Hypothesis: 6-hour strategy using daily context with Camarilla pivot levels.
Long when price closes above daily R3 with volume > 1.5x average and price > daily EMA200 (bullish trend).
Short when price closes below daily S3 with volume > 1.5x average and price < daily EMA200 (bearish trend).
Exit when price returns to daily pivot or volume drops below average.
Uses discrete position sizing (0.25) to minimize churn. Target: 15-25 trades/year.
Works in bull/bear by using EMA200 trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    if len(high) < 1:
        return np.full(len(high), np.nan), np.full(len(high), np.nan), \
               np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    pivot = (high + low + close) / 3.0
    range_val = high - low
    
    S3 = pivot - (range_val * 1.1 / 4)
    S4 = pivot - (range_val * 1.1 / 2)
    R3 = pivot + (range_val * 1.1 / 4)
    R4 = pivot + (range_val * 1.1 / 2)
    
    return S3, S4, R3, R4

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
    
    # Get daily data for context and Pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    S3_1d, S4_1d, R3_1d, R4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate daily EMA for trend filter
    ema_200_1d = calculate_ema(close_1d, 200)
    
    # Align indicators to 6-hour timeframe
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
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
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        s3 = S3_1d_aligned[i]
        r3 = R3_1d_aligned[i]
        pivot = pivot_1d_aligned[i]
        trend_up_1d = price > ema_200_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price returns to daily pivot or volume drops below average
            if price <= pivot or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to daily pivot or volume drops below average
            if price >= pivot or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above daily R3 with volume expansion and uptrend on daily
            if price > r3 and vol_ratio > 1.5 and trend_up_1d:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below daily S3 with volume expansion and downtrend on daily
            elif price < s3 and vol_ratio > 1.5 and not trend_up_1d:
                position = -1
                signals[i] = -0.25
    
    return signals