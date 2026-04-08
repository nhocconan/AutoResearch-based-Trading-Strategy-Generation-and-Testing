#!/usr/bin/env python3
"""
12h_1d1w_camarilla_pivot_v1
Hypothesis: Use weekly and daily Camarilla pivot levels on 12h chart with volume confirmation.
- Long when price touches or crosses above daily H3 level with volume expansion
- Short when price touches or crosses below daily L3 level with volume expansion
- Use weekly trend filter (price above/below weekly EMA20) to avoid counter-trend trades
- Designed for low trade frequency (12-37/year) to minimize fee drag
- Works in bull/bear via trend filter and volatility-based entry conditions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d1w_camarilla_pivot_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    if len(high) < 1:
        return np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    pivot = (high + low + close) / 3.0
    range_val = high - low
    
    H3 = pivot + (range_val * 1.1 / 4)
    L3 = pivot - (range_val * 1.1 / 4)
    
    return H3, L3

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
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_H3, camarilla_L3 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = calculate_ema(close_1w, 20)
    
    # Align indicators to 12h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        H3 = camarilla_H3_aligned[i]
        L3 = camarilla_L3_aligned[i]
        trend_up = price > ema_20_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price closes below L3 or volume drops significantly
            if price < L3 or vol_ratio < 0.7:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price closes above H3 or volume drops significantly
            if price > H3 or vol_ratio < 0.7:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches/crosses above H3 with volume expansion and uptrend
            if price >= H3 and vol_ratio > 1.8 and trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches/crosses below L3 with volume expansion and downtrend
            elif price <= L3 and vol_ratio > 1.8 and not trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals