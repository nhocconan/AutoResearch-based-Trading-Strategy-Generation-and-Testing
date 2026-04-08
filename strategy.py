#!/usr/bin/env python3
"""
4h_12h_camarilla_pivot_v1
Hypothesis: 4-hour strategy using 12-hour Camarilla pivot with volume confirmation and trend filter.
Long when price breaks above 12h R1 with volume > 2.5x average and price > 12h EMA50.
Short when price breaks below 12h S1 with volume > 2.5x average and price < 12h EMA50.
Exit when price crosses opposite 12h level OR volume falls below 1.5x average.
Uses 12h timeframe for better signal quality in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_v1"
timeframe = "4h"
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
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Pivot (using previous 12h bar's data)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    # 12h support/resistance levels (Standard Camarilla S1/R1)
    S1_12h = pivot_12h - (range_12h * 1.1 / 12)  # 12h S1
    R1_12h = pivot_12h + (range_12h * 1.1 / 12)  # 12h R1
    
    # Calculate 12h EMA for trend filter
    ema_50_12h = calculate_ema(close_12h, 50)
    
    # Align indicators to 4-hour timeframe
    S1_12h_aligned = align_htf_to_ltf(prices, df_12h, S1_12h)
    R1_12h_aligned = align_htf_to_ltf(prices, df_12h, R1_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(S1_12h_aligned[i]) or np.isnan(R1_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        S1 = S1_12h_aligned[i]
        R1 = R1_12h_aligned[i]
        trend_up_12h = price > ema_50_12h_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 12h S1 or volume drops below 1.5x average
            if price < S1 or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 12h R1 or volume drops below 1.5x average
            if price > R1 or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 12h R1 with volume expansion and uptrend on 12h
            if price > R1 and vol_ratio > 2.5 and trend_up_12h:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 12h S1 with volume expansion and downtrend on 12h
            elif price < S1 and vol_ratio > 2.5 and not trend_up_12h:
                position = -1
                signals[i] = -0.25
    
    return signals