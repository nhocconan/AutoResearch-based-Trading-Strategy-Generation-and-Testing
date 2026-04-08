# 6h_12h1d_camarilla_pivot_v1
# Hypothesis: 6-hour strategy using 12-hour and 1-day context with Camarilla pivot levels.
# Long when price breaks above 12h R4 with 1d trend filter and volume confirmation.
# Short when price breaks below 12h S4 with 1d trend filter and volume confirmation.
# Uses 12h Camarilla levels for structure and 1d EMA200 for trend filter.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h1d_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for a series of high/low/close"""
    if len(high) < 1:
        return (np.full(len(high), np.nan), np.full(len(high), np.nan),
                np.full(len(high), np.nan), np.full(len(high), np.nan))
    
    pivot = (high + low + close) / 3.0
    range_val = high - low
    
    # Camarilla levels
    H3 = pivot + (range_val * 1.1 / 4)
    L3 = pivot - (range_val * 1.1 / 4)
    H4 = pivot + (range_val * 1.1 / 2)
    L4 = pivot - (range_val * 1.1 / 2)
    
    return H3, L3, H4, L4

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
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (using 12h data)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    H3_12h, L3_12h, H4_12h, L4_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = calculate_ema(close_1d, 200)
    
    # Align indicators to 6h timeframe
    H4_12h_aligned = align_htf_to_ltf(prices, df_12h, H4_12h)
    L4_12h_aligned = align_htf_to_ltf(prices, df_12h, L4_12h)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(H4_12h_aligned[i]) or np.isnan(L4_12h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        H4 = H4_12h_aligned[i]
        L4 = L4_12h_aligned[i]
        trend_up_1d = price > ema_200_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price breaks below 12h L4 or volume drops below average
            if price < L4 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above 12h H4 or volume drops below average
            if price > H4 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 12h H4 with volume expansion and uptrend on daily
            if price > H4 and vol_ratio > 1.8 and trend_up_1d:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 12h L4 with volume expansion and downtrend on daily
            elif price < L4 and vol_ratio > 1.8 and not trend_up_1d:
                position = -1
                signals[i] = -0.25
    
    return signals