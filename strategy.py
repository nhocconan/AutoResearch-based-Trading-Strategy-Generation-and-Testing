#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_Filter
# Hypothesis: KAMA (adaptive trend) + RSI (momentum) + Choppiness Index (regime) captures trend with momentum in favorable regimes.
# KAMA filters noise, RSI avoids overextended entries, Chop filter avoids ranging markets.
# Works in bull (KAMA up + RSI > 50 + Chop < 61.8) and bear (KAMA down + RSI < 50 + Chop < 61.8).
# Target: 15-25 trades/year per symbol.

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    # Fix dimensions: change length n-10, volatility length n-1
    er = np.zeros(n)
    er[10:] = change / np.where(volatility[9:] == 0, 1, volatility[9:])  # avoid div by zero
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros(n)
    kama[:10] = close[:10]
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length
    rsi_padded = np.full(n, np.nan)
    rsi_padded[14:] = rsi
    
    # Calculate Choppiness Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop formula: 100 * log10(sum_tr / (hh - ll)) / log10(14)
    chop = 100 * np.log10(sum_tr / np.where(hh - ll == 0, 1, hh - ll)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # KAMA needs 10, RSI needs 14, Chop needs 14
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_padded[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up (trend up), RSI > 50 (bullish momentum), Chop < 61.8 (trending market)
            if (close[i] > kama[i] and 
                rsi_padded[i] > 50 and 
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down (trend down), RSI < 50 (bearish momentum), Chop < 61.8 (trending market)
            elif (close[i] < kama[i] and 
                  rsi_padded[i] < 50 and 
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend fails OR momentum fades OR market becomes choppy
            if (close[i] <= kama[i] or 
                rsi_padded[i] <= 50 or 
                chop[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend fails OR momentum fades OR market becomes choppy
            if (close[i] >= kama[i] or 
                rsi_padded[i] >= 50 or 
                chop[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals