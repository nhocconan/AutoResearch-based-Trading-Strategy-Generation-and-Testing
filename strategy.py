#!/usr/bin/env python3
"""
4h KAMA + RSI + Chop Regime Filter
Long: KAMA trend up AND RSI(14) crosses above 50 AND Chop > 61.8 (trending regime)
Short: KAMA trend down AND RSI(14) crosses below 50 AND Chop > 61.8 (trending regime)
Exit: Opposite RSI cross OR Chop < 38.2 (range regime)
Uses KAMA for adaptive trend, RSI for momentum confirmation, Chop for regime filter
Target: 20-40 trades/year per symbol (80-160 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA and Chop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA (ER=10) on 1d close
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # Calculate Chop (14) on 1d
    atr_1d = np.zeros_like(close_1d)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close_1d[:-1]))
    atr_1d[1:] = tr2
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(np.maximum.accumulate(high)).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(np.minimum.accumulate(low)).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, 14)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(rsi[i-1])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        chop_val = chop_aligned[i]
        rsi_val = rsi[i]
        rsi_prev = rsi[i-1]
        
        if position == 0:
            # Long: KAMA up AND RSI crosses above 50 AND Chop > 61.8 (trending)
            if kama_val > close[i] and rsi_prev <= 50 and rsi_val > 50 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down AND RSI crosses below 50 AND Chop > 61.8 (trending)
            elif kama_val < close[i] and rsi_prev >= 50 and rsi_val < 50 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses below 50 OR Chop < 38.2 (range)
            if rsi_prev >= 50 and rsi_val < 50 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses above 50 OR Chop < 38.2 (range)
            if rsi_prev <= 50 and rsi_val > 50 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_ChopRegime"
timeframe = "4h"
leverage = 1.0