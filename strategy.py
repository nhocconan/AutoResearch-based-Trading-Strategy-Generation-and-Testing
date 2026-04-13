#!/usr/bin/env python3
"""
4h_1D_KAMA_RSI_Chop_Filter
Hypothesis: Use daily KAMA direction for trend bias, RSI(2) for mean-reversion entries, and Choppiness Index to avoid whipsaws. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).rolling(window=er_length, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, length=2):
    """Calculate RSI."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_chop(high, low, close, length=14):
    """Calculate Choppiness Index."""
    atr = np.abs(high - low)
    atr_sum = pd.Series(atr).rolling(window=length, min_periods=1).sum()
    maxh = pd.Series(high).rolling(window=length, min_periods=1).max()
    minl = pd.Series(low).rolling(window=length, min_periods=1).min()
    chop = 100 * np.log10(atr_sum / (maxh - minl)) / np.log10(length)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d indicators
    kama_1d = calculate_kama(close_1d)
    chop_1d = calculate_chop(high_1d, low_1d, close_1d)
    
    # Align to 4h
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h RSI(2)
    rsi_2 = calculate_rsi(close)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(rsi_2[i])):
            signals[i] = 0.0
            continue
        
        # Long: price > KAMA (uptrend) + RSI < 10 (oversold) + Chop > 61.8 (ranging)
        long_condition = (close[i] > kama_1d_aligned[i]) and (rsi_2[i] < 10) and (chop_1d_aligned[i] > 61.8)
        
        # Short: price < KAMA (downtrend) + RSI > 90 (overbought) + Chop > 61.8 (ranging)
        short_condition = (close[i] < kama_1d_aligned[i]) and (rsi_2[i] > 90) and (chop_1d_aligned[i] > 61.8)
        
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

name = "4h_1D_KAMA_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0