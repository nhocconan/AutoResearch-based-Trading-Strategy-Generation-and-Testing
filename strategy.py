#!/usr/bin/env python3
"""
4h_1d_kama_rsi_chop
Uses KAMA on 1d timeframe to identify trend direction.
Enters on 4h when price crosses above/below KAMA with RSI confirmation and chop filter.
Exits when price crosses back below/above KAMA.
Kaufman Adaptive Moving Average adapts to market noise - faster in trends, slower in chop.
Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
Works in trending markets by following KAMA direction, avoids chop via Choppiness Index filter.
"""

name = "4h_1d_kama_rsi_chop"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1))**2
    kama = np.full_like(close, np.nan, dtype=float)
    kama[er_period] = close[er_period]
    for i in range(er_period+1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def choppiness_index(high, low, close, period=14):
    """Choppiness Index: higher = choppy/ranging, lower = trending"""
    atr = np.zeros_like(close)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr[1:] = tr
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    range_max_min = highest_high - lowest_low
    cpi = np.where(range_max_min != 0, 
                   100 * np.log10(atr_sum / range_max_min) / np.log10(period),
                   50)
    return cpi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA on daily
    kama_1d = kama(close_1d, er_period=10, fast_period=2, slow_period=30)
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # RSI on 4h (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index on 4h (14-period) - avoid choppy markets
    cpi = choppiness_index(high, low, close, period=14)
    chop_filter = cpi < 61.8  # Only trade when NOT choppy (trending market)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(kama_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: price above KAMA, RSI > 50 (bullish momentum), not choppy
        if (close[i] > kama_aligned[i] and rsi[i] > 50 and chop_filter[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price below KAMA, RSI < 50 (bearish momentum), not choppy
        elif (close[i] < kama_aligned[i] and rsi[i] < 50 and chop_filter[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price crosses back below/above KAMA
        elif position == 1 and close[i] < kama_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > kama_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals