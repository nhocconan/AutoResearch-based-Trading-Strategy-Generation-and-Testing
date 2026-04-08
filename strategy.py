#!/usr/bin/env python3
"""
1h_4h1d_trend_follow_v1
Hypothesis: 1h trend following with 4h/1d filters to reduce noise.
- Long when 1h EMA12 > EMA26 AND 4h EMA50 > EMA100 AND 1d EMA50 > EMA200 AND volume > 1.5x avg
- Short when 1h EMA12 < EMA26 AND 4h EMA50 < EMA100 AND 1d EMA50 < EMA200 AND volume > 1.5x avg
- Exit when 1h EMA crosses back or volume drops below average
- Uses 4h/1d for trend direction, 1h for entry timing
- Target: 20-40 trades/year to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_follow_v1"
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
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate EMAs
    ema_12_1h = calculate_ema(close, 12)
    ema_26_1h = calculate_ema(close, 26)
    
    close_4h = df_4h['close'].values
    ema_50_4h = calculate_ema(close_4h, 50)
    ema_100_4h = calculate_ema(close_4h, 100)
    
    close_1d = df_1d['close'].values
    ema_50_1d = calculate_ema(close_1d, 50)
    ema_200_1d = calculate_ema(close_1d, 200)
    
    # Align HTF indicators
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_100_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_100_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_12_1h[i]) or np.isnan(ema_26_1h[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_100_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        ema_fast_1h = ema_12_1h[i] > ema_26_1h[i]
        ema_fast_4h = ema_50_4h_aligned[i] > ema_100_4h_aligned[i]
        ema_fast_1d = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: EMA cross down or volume drops
            if not ema_fast_1h or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: EMA cross up or volume drops
            if ema_fast_1h or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: all EMAs aligned up + volume
            if ema_fast_1h and ema_fast_4h and ema_fast_1d and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.20
            # Enter short: all EMAs aligned down + volume
            elif not ema_fast_1h and not ema_fast_4h and not ema_fast_1d and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.20
    
    return signals