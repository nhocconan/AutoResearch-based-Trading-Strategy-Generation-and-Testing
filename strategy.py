# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_1dKAMA_RSI_Chop_Filter
KAMA trend direction + RSI momentum + Choppiness regime filter.
Long when KAMA rising, RSI > 50, and choppy market (CHOP > 61.8).
Short when KAMA falling, RSI < 50, and choppy market (CHOP > 61.8).
Uses daily timeframe for KAMA and RSI, 12h for execution.
Designed to work in both bull and bear markets by avoiding strong trends.
"""

name = "12h_1dKAMA_RSI_Chop_Filter"
timeframe = "12h"
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
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily KAMA (ER=10, fast=2, slow=30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # Will be calculated properly below
    
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        direction = abs(close_1d[i] - close_1d[i-9]) if i >= 9 else abs(close_1d[i] - close_1d[0])
        volatility_sum = np.sum(np.abs(np.diff(close_1d[max(0,i-9):i+1]))) if i >= 9 else np.sum(np.abs(np.diff(close_1d[:i+1])))
        er[i] = direction / volatility_sum if volatility_sum > 0 else 0
    
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily Choppiness Index(14)
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(np.diff(high))
    tr2 = np.abs(np.diff(low))
    tr3 = np.abs(np.diff(close_1d))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if atr[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(atr[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    # Align indicators to 12h timeframe
    kama_12h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_12h[i]) or np.isnan(rsi_12h[i]) or np.isnan(chop_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, choppy market (CHOP > 61.8)
            if kama_12h[i] > kama_12h[i-1] and rsi_12h[i] > 50 and chop_12h[i] > 61.8:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: KAMA falling, RSI < 50, choppy market (CHOP > 61.8)
            elif kama_12h[i] < kama_12h[i-1] and rsi_12h[i] < 50 and chop_12h[i] > 61.8:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: KAMA falling or RSI < 50 or chop < 61.8 (trending market)
            if bars_since_entry >= 2 and (kama_12h[i] < kama_12h[i-1] or rsi_12h[i] < 50 or chop_12h[i] < 61.8):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA rising or RSI > 50 or chop < 61.8 (trending market)
            if bars_since_entry >= 2 and (kama_12h[i] > kama_12h[i-1] or rsi_12h[i] > 50 or chop_12h[i] < 61.8):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals