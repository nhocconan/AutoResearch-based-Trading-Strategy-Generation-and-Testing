#!/usr/bin/env python3
"""
1h_4d_rsi_mean_reversion_v1
Hypothesis: 1-hour strategy using 4-hour RSI for mean reversion with daily trend filter.
Long when 4h RSI < 30 (oversold) and price > daily EMA50 (bullish trend).
Short when 4h RSI > 70 (overbought) and price < daily EMA50 (bearish trend).
Exit when RSI returns to neutral (40-60 range).
Uses 4h for signal direction, 1h only for entry timing. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_rsi_mean_reversion_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    
    # Get 4h data for RSI signal
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h RSI
    close_4h = df_4h['close'].values
    rsi_4h = calculate_rsi(close_4h, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50_1d = calculate_ema(close_1d, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        rsi = rsi_4h_aligned[i]
        price = close[i]
        ema_50 = ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: RSI returns to neutral (>40) or trend changes
            if rsi > 40 or price < ema_50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: RSI returns to neutral (<60) or trend changes
            if rsi < 60 or price > ema_50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: 4h RSI oversold (<30) and bullish trend (price > daily EMA50)
            if rsi < 30 and price > ema_50:
                position = 1
                signals[i] = 0.20
            # Enter short: 4h RSI overbought (>70) and bearish trend (price < daily EMA50)
            elif rsi > 70 and price < ema_50:
                position = -1
                signals[i] = -0.20
    
    return signals