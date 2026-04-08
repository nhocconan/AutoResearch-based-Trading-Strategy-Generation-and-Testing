#!/usr/bin/env python3
"""
1h_4h1d_rsi_ema_v1
Hypothesis: 1-hour strategy using 4-hour RSI(14) and 1-day EMA(50) for trend direction,
with 1-hour RSI pullback entries. Long when 4h RSI > 50, price > 1d EMA50, and 1h RSI < 40
pullback from oversold. Short when 4h RSI < 50, price < 1d EMA50, and 1h RSI > 60 pullback
from overbought. Uses higher timeframe for signal direction to reduce whipsaw, lower timeframe
for precise entry timing. Target: 15-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_rsi_ema_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI with proper handling"""
    if len(close) < period + 1:
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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4-hour and 1-day data for context
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    rsi_4h = calculate_rsi(df_4h['close'].values, 14)
    
    # Calculate 1d EMA(50)
    ema_50_1d = np.full_like(df_1d['close'].values, np.nan, dtype=float)
    if len(df_1d) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1d[49] = np.mean(df_1d['close'].values[:50])
        for i in range(50, len(df_1d)):
            ema_50_1d[i] = alpha * df_1d['close'].values[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Calculate 1h RSI(14) for entry timing
    rsi_1h = calculate_rsi(close, 14)
    
    # Align indicators to 1-hour timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_1h[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        rsi4 = rsi_4h_aligned[i]
        ema1d = ema_50_1d_aligned[i]
        rsi1 = rsi_1h[i]
        price = close[i]
        
        if position == 1:  # Long
            # Exit: 4h RSI < 40 or price < 1d EMA50
            if rsi4 < 40 or price < ema1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: 4h RSI > 60 or price > 1d EMA50
            if rsi4 > 60 or price > ema1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: 4h RSI > 50 (uptrend), price > 1d EMA50, and 1h RSI < 40 pullback
            if rsi4 > 50 and price > ema1d and rsi1 < 40:
                position = 1
                signals[i] = 0.20
            # Enter short: 4h RSI < 50 (downtrend), price < 1d EMA50, and 1h RSI > 60 pullback
            elif rsi4 < 50 and price < ema1d and rsi1 > 60:
                position = -1
                signals[i] = -0.20
    
    return signals