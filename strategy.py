#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_v2
Hypothesis: 4-hour Donchian breakout with daily trend filter and volume confirmation.
Long when price breaks above 4h Donchian(20) high with volume > 2x average and price > daily EMA200.
Short when price breaks below 4h Donchian(20) low with volume > 2x average and price < daily EMA200.
Exit when price crosses opposite 4h Donchian(10) level or volume falls below 1.5x average.
Designed to work in both bull (trend following) and bear (mean reversion via Donchian reversal).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_v2"
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
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = calculate_ema(close_1d, 200)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h Donchian channels
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    high_10 = np.full(n, np.nan)
    low_10 = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 20:
            high_20[i] = np.max(high[i-20:i])
            low_20[i] = np.min(low[i-20:i])
        if i >= 10:
            high_10[i] = np.max(high[i-10:i])
            low_10[i] = np.min(low[i-10:i])
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(high_10[i]) or np.isnan(low_10[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        trend_up = price > ema_200_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 4h Donchian(10) low or volume drops
            if price < low_10[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 4h Donchian(10) high or volume drops
            if price > high_10[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 4h Donchian(20) high with volume expansion and uptrend
            if price > high_20[i] and vol_ratio > 2.0 and trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 4h Donchian(20) low with volume expansion and downtrend
            elif price < low_20[i] and vol_ratio > 2.0 and not trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals