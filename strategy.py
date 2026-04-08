#!/usr/bin/env python3
"""
4h_12h1d_camarilla_pivot_v4
Hypothesis: Further refined 12h/1d Camarilla pivot strategy with stricter entry criteria to reduce trade frequency.
- Long when price crosses above Camarilla H3 (12h) with volume > 2.5x avg, price > 1d EMA50 (trend filter)
- Short when price crosses below Camarilla L3 (12h) with volume > 2.5x avg, price < 1d EMA50 (trend filter)
- Exit when price crosses opposite H3/L3 level or volume drops below average
- Uses discrete position sizing (0.20) to minimize churn
- Designed for 10-20 trades/year to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h1d_camarilla_pivot_v4"
timeframe = "4h"
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
    
    # Get 12-hour data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_H3_12h, camarilla_L3_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Calculate 1-day EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = calculate_ema(close_1d, 50)
    
    # Align indicators to 4-hour timeframe
    camarilla_H3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_H3_12h)
    camarilla_L3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_L3_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_H3_12h_aligned[i]) or np.isnan(camarilla_L3_12h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        H3 = camarilla_H3_12h_aligned[i]
        L3 = camarilla_L3_12h_aligned[i]
        trend_up_1d = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price closes below L3 or volume drops below average
            if price < L3 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: price closes above H3 or volume drops below average
            if price > H3 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price crosses above H3 with volume expansion and uptrend on 1d
            if price > H3 and vol_ratio > 2.5 and trend_up_1d:
                position = 1
                signals[i] = 0.20
            # Enter short: price crosses below L3 with volume expansion and downtrend on 1d
            elif price < L3 and vol_ratio > 2.5 and not trend_up_1d:
                position = -1
                signals[i] = -0.20
    
    return signals