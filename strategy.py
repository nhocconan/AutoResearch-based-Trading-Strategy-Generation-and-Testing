#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA (Kaufman Adaptive Moving Average) with RSI and Choppiness Index regime filter.
# KAMA adapts to market conditions: fast in trends, slow in ranges. RSI identifies overbought/oversold.
# Choppiness Index (CHOP) filters ranging markets (CHOP > 61.8) where we avoid trend following.
# Long when KAMA upward, RSI > 50, and CHOP < 61.8 (trending market).
# Short when KAMA downward, RSI < 50, and CHOP < 61.8.
# Works in both bull and bear by following adaptive trend with regime filter to avoid whipsaw.
name = "12h_KAMA_RSI_Chop_Regime"
timeframe = "12h"
leverage = 1.0

def kama(close, length=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    if len(close) < length:
        return np.full_like(close, np.nan, dtype=np.float64)
    dir = np.abs(close - np.roll(close, length))
    vol = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    er = np.where(vol != 0, dir / vol, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[length-1] = close[length-1]
    for i in range(length, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def rsi(close, length=14):
    """Relative Strength Index"""
    if len(close) < length + 1:
        return np.full_like(close, np.nan, dtype=np.float64)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan, dtype=np.float64)
    avg_loss = np.full_like(close, np.nan, dtype=np.float64)
    avg_gain[length] = np.mean(gain[:length])
    avg_loss[length] = np.mean(loss[:length])
    for i in range(length+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
        avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def choppiness_index(high, low, close, length=14):
    """Choppiness Index: identifies ranging vs trending markets"""
    if len(close) < length:
        return np.full_like(close, np.nan, dtype=np.float64)
    atr = np.zeros_like(close)
    for i in range(1, len(close)):
        atr[i] = max(
            high[i] - low[i],
            np.abs(high[i] - close[i-1]),
            np.abs(low[i] - close[i-1])
        )
    sum_atr = np.zeros_like(close)
    for i in range(length, len(close)):
        sum_atr[i] = np.sum(atr[i-length+1:i+1])
    hh = np.zeros_like(close)
    ll = np.zeros_like(close)
    for i in range(length-1, len(close)):
        hh[i] = np.max(high[i-length+1:i+1])
        ll[i] = np.min(low[i-length+1:i+1])
    chop = np.where((hh - ll) != 0, 100 * np.log10(sum_atr / (hh - ll)) / np.log10(length), 50)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA, RSI, and Choppiness Index
    kama_val = kama(close, length=10, fast=2, slow=30)
    rsi_val = rsi(close, length=14)
    chop = choppiness_index(high, low, close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA upward (current > previous), RSI > 50, CHOP < 61.8 (trending)
            if kama_val[i] > kama_val[i-1] and rsi_val[i] > 50 and chop[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA downward (current < previous), RSI < 50, CHOP < 61.8 (trending)
            elif kama_val[i] < kama_val[i-1] and rsi_val[i] < 50 and chop[i] < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA downward or RSI < 50
            if kama_val[i] < kama_val[i-1] or rsi_val[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA upward or RSI > 50
            if kama_val[i] > kama_val[i-1] or rsi_val[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals