#!/usr/bin/env python3
"""
6h_12h1d_camarilla_pivot_v1
Hypothesis: 6-hour strategy using 12h and 1d Camarilla pivot with volume confirmation.
Long when price breaks above 12h R2 with volume > 2x average and price > 1d EMA50.
Short when price breaks below 12h S2 with volume > 2x average and price < 1d EMA50.
Exit when price crosses opposite 12h level OR volume falls below 1.5x average.
Uses higher timeframe pivots for better signal quality in both bull and bear markets.
Target: 15-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h1d_camarilla_pivot_v1"
timeframe = "6h"
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
    
    # Get 12-hour and 1-day data for context
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Pivot (using previous 12h bar's data)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    # 12h support/resistance levels (Camarilla S2/R2)
    S2_12h = pivot_12h - (range_12h * 1.1 / 6)  # 12h S2
    R2_12h = pivot_12h + (range_12h * 1.1 / 6)  # 12h R2
    
    # Calculate 1d EMA for trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    
    # Align indicators to 6-hour timeframe
    S2_12h_aligned = align_htf_to_ltf(prices, df_12h, S2_12h)
    R2_12h_aligned = align_htf_to_ltf(prices, df_12h, R2_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(S2_12h_aligned[i]) or np.isnan(R2_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        S2 = S2_12h_aligned[i]
        R2 = R2_12h_aligned[i]
        trend_up_1d = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 12h S2 or volume drops below 1.5x average
            if price < S2 or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 12h R2 or volume drops below 1.5x average
            if price > R2 or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 12h R2 with volume expansion and uptrend on 1d
            if price > R2 and vol_ratio > 2.0 and trend_up_1d:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 12h S2 with volume expansion and downtrend on 1d
            elif price < S2 and vol_ratio > 2.0 and not trend_up_1d:
                position = -1
                signals[i] = -0.25
    
    return signals