#!/usr/bin/env python3
"""
6h_1w21d_fib_ext_v1
Hypothesis: Use weekly Fibonacci extensions from weekly swing points with 21-day EMA filter for 6h timeframe.
- Long when price breaks above weekly 1.618 extension with volume > 1.8x average and price > 21-day EMA
- Short when price breaks below weekly 0.618 extension with volume > 1.8x average and price < 21-day EMA
- Exit when price crosses back below/above the weekly VWAP or volume drops below average
- Uses discrete position sizing (0.25) to minimize churn
- Designed for 15-30 trades/year to avoid fee drag in 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w21d_fib_ext_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fib_extension(high, low, close):
    """Calculate weekly Fibonacci extension levels (0.618 and 1.618)"""
    if len(high) < 2:
        return np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    # Find swing high and low (simplified: use max/min of period)
    period_high = np.max(high)
    period_low = np.min(low)
    diff = period_high - period_low
    
    # Fibonacci extension levels
    fib_0618 = period_low + diff * 0.618  # Retracement level for shorts
    fib_1618 = period_high + diff * 0.618  # Extension level for longs
    
    return np.full(len(high), fib_1618), np.full(len(high), fib_0618)

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP"""
    if len(volume) == 0 or np.sum(volume) == 0:
        return np.full(len(close), np.nan)
    
    typical_price = (high + low + close) / 3.0
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    return vwap

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
    
    # Get weekly data for Fibonacci levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for VWAP and EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate weekly Fibonacci extension levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    fib_ext_1618, fib_ext_0618 = calculate_fib_extension(high_1w, low_1w, close_1w)
    
    # Calculate daily VWAP
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    vwap_1d = calculate_vwap(high_1d, low_1d, close_1d, volume_1d)
    
    # Calculate daily 21-period EMA for trend filter
    ema_21_1d = calculate_ema(close_1d, 21)
    
    # Align indicators to 6-hour timeframe
    fib_ext_1618_aligned = align_htf_to_ltf(prices, df_1w, fib_ext_1618)
    fib_ext_0618_aligned = align_htf_to_ltf(prices, df_1w, fib_ext_0618)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(fib_ext_1618_aligned[i]) or np.isnan(fib_ext_0618_aligned[i]) or
            np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_21_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        fib_up = fib_ext_1618_aligned[i]
        fib_down = fib_ext_0618_aligned[i]
        vwap = vwap_1d_aligned[i]
        ema_21 = ema_21_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below VWAP or volume drops below average
            if price < vwap or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above VWAP or volume drops below average
            if price > vwap or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly 1.618 extension with volume expansion and above 21-day EMA
            if price > fib_up and vol_ratio > 1.8 and price > ema_21:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly 0.618 extension with volume expansion and below 21-day EMA
            elif price < fib_down and vol_ratio > 1.8 and price < ema_21:
                position = -1
                signals[i] = -0.25
    
    return signals