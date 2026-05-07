#!/usr/bin/env python3
# 12h_1dKAMA_RSI_ChopFilter
# Hypothesis: 12h KAMA direction + RSI(14) + daily Choppiness Index filter captures
# trending moves while avoiding chop, effective in both bull and bear markets.
# Uses daily timeframe for regime filter to reduce false signals.
# Target: 20-30 trades/year to minimize fee decay.

name = "12h_1dKAMA_RSI_ChopFilter"
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
    
    # Get 12h data for KAMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 12h close
    close_12h = df_12h['close'].values
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.abs(np.diff(close_12h))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * 0.29 + 0.06) ** 2
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h = kama
    
    # Get daily data for RSI and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate RSI(14) on daily close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(np.diff(high_1d))
    tr2 = np.abs(np.diff(low_1d))
    tr3 = np.abs(np.diff(close_1d))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0,
                    100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14),
                    50)
    # Handle NaN from sum
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Align all indicators to 12h timeframe
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, chop < 61.8 (trending)
            if close[i] > kama_12h_aligned[i] and rsi_aligned[i] > 50 and chop_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, chop < 61.8 (trending)
            elif close[i] < kama_12h_aligned[i] and rsi_aligned[i] < 50 and chop_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA or chop > 61.8 (choppy)
            if close[i] < kama_12h_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA or chop > 61.8 (choppy)
            if close[i] > kama_12h_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals