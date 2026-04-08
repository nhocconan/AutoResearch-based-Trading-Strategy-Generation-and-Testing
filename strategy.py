#!/usr/bin/env python3
# 12h_kama_rsi_chop
# Hypothesis: KAMA trend direction on 12h filtered by RSI momentum and Choppiness regime filter from 1d.
# Long when KAMA is rising (trending up), RSI < 40 (pullback in uptrend), and 1d Choppiness > 61.8 (range-bound market).
# Short when KAMA is falling (trending down), RSI > 60 (pullback in downtrend), and 1d Choppiness > 61.8.
# Uses mean-reversion within higher timeframe trend, optimized for choppy markets like 2025. Target: 15-30 trades/year (~60-120 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_rsi_chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # Get daily data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h KAMA (trend filter)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[30] = close[30]  # seed
    for i in range(31, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 12h RSI (momentum filter)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])  # align with close
    
    # Calculate 1d Choppiness Index (regime filter)
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(np.subtract(high_1d[1:], low_1d[1:]))
    tr2 = np.abs(np.subtract(high_1d[1:], close_1d[:-1]))
    tr3 = np.abs(np.subtract(low_1d[1:], close_1d[:-1]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align indices
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = np.zeros_like(close_1d)
    mask = (max_high - min_low) != 0
    chop[mask] = 100 * np.log10(np.sum(tr[-14:], axis=1) / np.log(10) / 14) / np.log10(max_high[mask] - min_low[mask])
    chop = np.concatenate([np.full(14, np.nan), chop])  # align with close_1d
    
    # Align 1d indicators to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR RSI > 50 (exit pullback)
            if (kama[i] < kama[i-1]) or (rsi[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR RSI < 50 (exit pullback)
            if (kama[i] > kama[i-1]) or (rsi[i] < 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Chop filter: market must be choppy (range-bound) for mean reversion
            chop_ok = chop_aligned[i] > 61.8
            
            # Long entry: KAMA rising (uptrend), RSI < 40 (oversold pullback), choppy market
            if (kama[i] > kama[i-1]) and (rsi[i] < 40) and chop_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: KAMA falling (downtrend), RSI > 60 (overbought pullback), choppy market
            elif (kama[i] < kama[i-1]) and (rsi[i] > 60) and chop_ok:
                position = -1
                signals[i] = -0.25
    
    return signals