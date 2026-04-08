#!/usr/bin/env python3
"""
1d_rsi_pullback_v1
Hypothesis: RSI pullback strategy on daily timeframe with strict entry criteria to limit trades.
- Long when RSI(14) < 30 (oversold) and price > EMA(50) (uptrend filter)
- Short when RSI(14) > 70 (overbought) and price < EMA(50) (downtrend filter)
- Exit when RSI returns to neutral range (40-60)
- Uses daily timeframe to naturally limit trades (target: 10-25/year)
- Designed for 15-25 trades/year to avoid fee drag
- Works in both bull and bear markets by capturing mean reversion within trends
"""

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

name = "1d_rsi_pullback_v1"
timeframe = "1d"
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
    
    # First average
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    # Subsequent averages
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
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
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter (optional, can be removed if too restrictive)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily RSI
    rsi = calculate_rsi(close, 14)
    
    # Calculate daily EMA50 for trend filter
    ema_50 = calculate_ema(close, 50)
    
    # Calculate weekly EMA200 for higher timeframe trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = calculate_ema(close_1w, 200)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if np.isnan(rsi[i]) or np.isnan(ema_50[i]) or np.isnan(ema_200_1w_aligned[i]):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        price = close[i]
        ema50_val = ema_50[i]
        ema200_1w_val = ema_200_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: RSI returns to neutral range (40-60) or price breaks below EMA50
            if rsi_val >= 40 and rsi_val <= 60 or price < ema50_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: RSI returns to neutral range (40-60) or price breaks above EMA50
            if rsi_val >= 40 and rsi_val <= 60 or price > ema50_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI oversold (<30) and price above both EMAs (uptrend)
            if rsi_val < 30 and price > ema50_val and price > ema200_1w_val:
                position = 1
                signals[i] = 0.25
            # Enter short: RSI overbought (>70) and price below both EMAs (downtrend)
            elif rsi_val > 70 and price < ema50_val and price < ema200_1w_val:
                position = -1
                signals[i] = -0.25
    
    return signals