#!/usr/bin/env python3
"""
12h_1d_kama_rsi_chop_v2
KAMA trend direction from daily timeframe + RSI mean reversion on 12h + chop filter.
Enters long when daily KAMA up, RSI < 30, and chop > 61.8 (ranging).
Enters short when daily KAMA down, RSI > 70, and chop > 61.8.
Exits when RSI crosses 50 or chop < 38.2 (trending).
Designed for low trade frequency to work in both bull and bear markets.
"""

name = "12h_1d_kama_rsi_chop_v2"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_length=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
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
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily KAMA for trend direction
    kama_1d = kama(close_1d, er_length=10, fast=2, slow=30)
    kama_up = kama_1d > np.roll(kama_1d, 1)
    kama_down = kama_1d < np.roll(kama_1d, 1)
    
    # Align KAMA signals to 12h
    kama_up_aligned = align_htf_to_ltf(prices, df_1d, kama_up.astype(float))
    kama_down_aligned = align_htf_to_ltf(prices, df_1d, kama_down.astype(float))
    
    # 12h RSI for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h Choppy Index for regime filter
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(np.roll(close, 1) - high)
    tr3 = np.abs(np.roll(close, 1) - low)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * np.sqrt(14) / (highest - lowest)) / np.log10(9)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama_up_aligned[i]) or np.isnan(kama_down_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: daily KAMA up, RSI oversold, choppy market
        if (kama_up_aligned[i] > 0.5 and rsi[i] < 30 and chop[i] > 61.8 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: daily KAMA down, RSI overbought, choppy market
        elif (kama_down_aligned[i] > 0.5 and rsi[i] > 70 and chop[i] > 61.8 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: RSI crosses 50 or market starts trending
        elif position == 1 and (rsi[i] > 50 or chop[i] < 38.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 50 or chop[i] < 38.2):
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