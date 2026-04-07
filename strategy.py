#!/usr/bin/env python3
"""
4h KAMA Direction + RSI + Chop Filter
Long when KAMA trending up, RSI > 50, and choppy market (Chop > 61.8)
Short when KAMA trending down, RSI < 50, and choppy market (Chop > 61.8)
Exit when conditions reverse
KAMA adapts to market noise, RSI filters momentum, Chop filter ensures mean-reversion environment
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_kama_rsi_chop_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === KAMA Calculation ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array lengths
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI Calculation ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])  # align with close
    
    # === Chop Calculation (using 1d HTF) ===
    df_1d = get_htf_data(prices, '1d')
    atr_1d = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        else:
            tr = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
        atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    # Max high - min low over 14 periods
    max_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    chop = 100 * np.log10(tr_sum / range_max_min) / np.log10(14)
    chop = np.concatenate([np.full(14, np.nan), chop[14:]])  # align
    
    # Align HTF Chop to LTF
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade in choppy markets (Chop > 61.8)
        if chop_aligned[i] <= 61.8:
            signals[i] = 0.0
            continue
        
        # KAMA direction: slope of KAMA
        if i >= 2:
            kama_slope = kama[i] - kama[i-2]
        else:
            kama_slope = 0
        
        # Entry conditions
        if kama_slope > 0 and rsi[i] > 50:
            # KAMA up and RSI bullish -> long
            signals[i] = 0.25
        elif kama_slope < 0 and rsi[i] < 50:
            # KAMA down and RSI bearish -> short
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals