#!/usr/bin/env python3
"""
12h_1d_kama_rsi_chop
Uses KAMA direction on 12h with RSI and chop regime filter.
- KAMA: Adaptive trend detection (ER=10)
- RSI(14): Overbought/Oversold signals
- Chop: Choppiness Index (14) to detect ranging vs trending
Logic:
  Long: KAMA rising + RSI < 30 (oversold) + Chop > 61.8 (ranging)
  Short: KAMA falling + RSI > 70 (overbought) + Chop > 61.8 (ranging)
Exit: Opposite signal or Chop < 38.2 (trending)
Designed for low trade frequency (<30/year) to minimize fee drag.
Works in both bull/bear by fading extremes in ranging markets.
"""

name = "12h_1d_kama_rsi_chop"
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

def rsi(close, length=14):
    """Relative Strength Index"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=length, min_periods=length).mean().values
    avg_loss = pd.Series(loss).rolling(window=length, min_periods=length).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi = np.concatenate([np.full(length, np.nan), rsi])
    return rsi

def choppiness_index(high, low, close, length=14):
    """Choppiness Index: 0 = trending, 100 = ranging"""
    atr = []
    for i in range(len(high)):
        tr = max(
            high[i] - low[i],
            np.abs(high[i] - close[i-1]) if i > 0 else 0,
            np.abs(low[i] - close[i-1]) if i > 0 else 0
        )
        atr.append(tr)
    atr = np.array(atr)
    sum_atr = pd.Series(atr).rolling(window=length, min_periods=length).sum().values
    hh = pd.Series(high).rolling(window=length, min_periods=length).max().values
    ll = pd.Series(low).rolling(window=length, min_periods=length).min().values
    chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(length)
    # Handle division by zero
    chop = np.where((hh - ll) == 0, 50, chop)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA on 12h
    kama_val = kama(close, er_length=10, fast=2, slow=30)
    kama_dir = np.diff(kama_val, prepend=kama_val[0])  # 1 if rising, -1 if falling
    
    # RSI on 12h
    rsi_val = rsi(close, length=14)
    
    # Chop on 12h
    chop_val = choppiness_index(high, low, close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or np.isnan(chop_val[i])):
            signals[i] = 0.0
            continue
        
        # Long: KAMA rising + RSI oversold + Chop > 61.8 (ranging)
        if (kama_dir[i] > 0 and rsi_val[i] < 30 and chop_val[i] > 61.8 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: KAMA falling + RSI overbought + Chop > 61.8 (ranging)
        elif (kama_dir[i] < 0 and rsi_val[i] > 70 and chop_val[i] > 61.8 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Opposite signal or Chop < 38.2 (trending)
        elif position == 1 and (kama_dir[i] < 0 or rsi_val[i] > 70 or chop_val[i] < 38.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (kama_dir[i] > 0 or rsi_val[i] < 30 or chop_val[i] < 38.2):
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