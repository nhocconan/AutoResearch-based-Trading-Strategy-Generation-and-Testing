#!/usr/bin/env python3
"""
1d KAMA with RSI and Chop Filter
Long when KAMA rising and RSI > 50 and Chop < 61.8 (trending up)
Short when KAMA falling and RSI < 50 and Chop < 61.8 (trending down)
Exit when RSI crosses 50 or Chop > 61.8 (range)
Designed to capture trends while avoiding choppy markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === KAMA (10) ===
    change = abs(np.diff(close, prepend=close[0]))
    volatility = abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Chop (14) ===
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    max_h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * 14 / (max_h - min_l)) / np.log10(14)
    chop = np.where((max_h - min_l) != 0, chop, 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI < 50 or Chop > 61.8
            if rsi[i] < 50 or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI > 50 or Chop > 61.8
            if rsi[i] > 50 or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trending up: KAMA rising and RSI > 50
            if kama[i] > kama[i-1] and rsi[i] > 50 and chop[i] < 61.8:
                position = 1
                signals[i] = 0.25
            # Trending down: KAMA falling and RSI < 50
            elif kama[i] < kama[i-1] and rsi[i] < 50 and chop[i] < 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals