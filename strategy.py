#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Trend_Filter_v2
Hypothesis: Use 1d KAMA to determine trend direction, combined with 1d RSI overbought/oversold conditions
and 1w trend filter to avoid counter-trend trades. Designed for low-frequency, high-conviction trades
that work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Calculate efficiency ratio
    er = np.zeros_like(close)
    for i in range(er_period, len(close)):
        dir_change = np.abs(close[i] - close[i-er_period])
        total_change = np.sum(change[i-er_period+1:i+1])
        er[i] = dir_change / (total_change + 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d KAMA
    kama_1d = calculate_kama(close_1d, er_period=10, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1d RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Simple 1w trend: price above/below 20-period EMA
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price > KAMA (uptrend), RSI < 30 (oversold), 1w price > EMA20 (bullish trend)
        long_condition = (close[i] > kama_1d_aligned[i]) and \
                         (rsi_1d_aligned[i] < 30) and \
                         (close[i] > ema_20_1w_aligned[i])
        
        # Short conditions: price < KAMA (downtrend), RSI > 70 (overbought), 1w price < EMA20 (bearish trend)
        short_condition = (close[i] < kama_1d_aligned[i]) and \
                          (rsi_1d_aligned[i] > 70) and \
                          (close[i] < ema_20_1w_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_KAMA_RSI_Trend_Filter_v2"
timeframe = "1d"
leverage = 1.0