#!/usr/bin/env python3
"""
4h_1d1w_donchian_breakout_v1
Hypothesis: 4-hour strategy using Donchian breakout with 1-day and 1-week context.
Long when price breaks above 4h Donchian(20) high with volume > 1.5x average and 1d EMA200 up.
Short when price breaks below 4h Donchian(20) low with volume > 1.5x average and 1d EMA200 down.
Exit when price crosses opposite Donchian level or volume drops below average.
Uses discrete position sizing (0.25) to minimize churn. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d1w_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    if len(high) < period:
        return np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    upper = np.full(len(high), np.nan)
    lower = np.full(len(high), np.nan)
    
    for i in range(period - 1, len(high)):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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
    
    # Get weekly data for additional context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels
    donchian_high, donchian_low = calculate_donchian(high, low, 20)
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = calculate_ema(close_1d, 200)
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = calculate_ema(close_1w, 50)
    
    # Align indicators to 4-hour timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, prices, donchian_high)  # Already LTF
    donchian_low_aligned = align_htf_to_ltf(prices, prices, donchian_low)   # Already LTF
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 50-period average
    vol_ma = np.full(n, np.nan)
    for i in range(50, n):
        vol_ma[i] = np.mean(volume[i-50:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        trend_up_1d = price > ema_200_1d_aligned[i]
        trend_up_1w = price > ema_50_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below Donchian low or volume drops below average
            if price < lower or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above Donchian high or volume drops below average
            if price > upper or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and uptrend on daily/weekly
            if price > upper and vol_ratio > 1.5 and trend_up_1d and trend_up_1w:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and downtrend on daily/weekly
            elif price < lower and vol_ratio > 1.5 and not trend_up_1d and not trend_up_1w:
                position = -1
                signals[i] = -0.25
    
    return signals