#!/usr/bin/env python3
"""
12h KAMA Direction with 1d RSI and 1w Chop Filter
Long when KAMA turns up and 1d RSI < 40 in low chop regime (trending down bounce)
Short when KAMA turns down and 1d RSI > 60 in low chop regime (trending up pullback)
Exit when KAMA reverses direction
Designed to catch trend exhaustion bounces in ranging markets with low volatility
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # === KAMA (10) ===
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * 0.6 + 0.06) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d RSI (14) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1w Chop (14) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    highest = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_1w = 100 * np.log10(highest - lowest) / np.log10(14) / np.log10(atr_1w.sum() / 14)
    chop_1w = np.where(atr_1w > 0, chop_1w, 50)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        if np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down
            if kama[i] < kama[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up
            if kama[i] > kama[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Low chop regime: chop < 38.2 (trending)
            if chop_1w_aligned[i] < 38.2:
                # KAMA turning up + RSI oversold
                if kama[i] > kama[i-1] and rsi_1d_aligned[i] < 40:
                    position = 1
                    signals[i] = 0.25
                # KAMA turning down + RSI overbought
                elif kama[i] < kama[i-1] and rsi_1d_aligned[i] > 60:
                    position = -1
                    signals[i] = -0.25
    
    return signals