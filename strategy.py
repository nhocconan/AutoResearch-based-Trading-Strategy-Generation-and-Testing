#!/usr/bin/env python3
"""
1h_4h1d_camarilla_pivot_v1
Hypothesis: Use daily Camarilla pivot (S1/R1) with 4h trend filter and 1h entry timing.
Long when price crosses above daily R1, volume > 2x average, and price > 4h EMA200.
Short when price crosses below daily S1, volume > 2x average, and price < 4h EMA200.
Exit when price crosses opposite level or volume drops below 1.5x average.
Uses discrete position sizing (0.20) to limit trades to ~15-35/year.
Designed for 1h timeframe with 4h/1d higher timeframe context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_camarilla_pivot_v1"
timeframe = "1h"
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
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context and Pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily Pivot (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # Daily support/resistance levels (Standard Camarilla S1/R1)
    S1_1d = pivot_1d - (range_1d * 1.1 / 12)  # Daily S1
    R1_1d = pivot_1d + (range_1d * 1.1 / 12)  # Daily R1
    
    # Calculate 4h EMA for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_200_4h = calculate_ema(close_4h, 200)
    
    # Align indicators to 1h timeframe
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(S1_1d_aligned[i]) or np.isnan(R1_1d_aligned[i]) or 
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        S1 = S1_1d_aligned[i]
        R1 = R1_1d_aligned[i]
        trend_up_4h = price > ema_200_4h_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below daily S1 or volume drops below 1.5x average
            if price < S1 or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: price crosses above daily R1 or volume drops below 1.5x average
            if price > R1 or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price crosses above daily R1 with volume expansion and uptrend on 4h
            if price > R1 and vol_ratio > 2.0 and trend_up_4h:
                position = 1
                signals[i] = 0.20
            # Enter short: price crosses below daily S1 with volume expansion and downtrend on 4h
            elif price < S1 and vol_ratio > 2.0 and not trend_up_4h:
                position = -1
                signals[i] = -0.20
    
    return signals