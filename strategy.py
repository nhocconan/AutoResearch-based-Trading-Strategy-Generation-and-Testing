#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: KAMA(10) trend direction + RSI(14) mean reversion + Choppiness Index regime filter.
In bull: long when KAMA up + RSI<30; in bear: short when KAMA down + RSI>70.
Chop filter avoids whipsaws in ranging markets (CHOP>61.8). Works in all regimes.
Target: 25-35 trades/year per symbol (100-140 over 4 years).
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
    volume = prices['volume'].values
    
    # KAMA trend direction (10-period)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)
    kama_dir[0] = 1
    
    # RSI(14) for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) for regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ok = volume[i] > 1.2 * vol_ma[i]
        chop_ok = chop[i] < 61.8  # Trending regime
        
        if position == 0:
            # Long: KAMA up + RSI oversold + volume + trending
            if kama_dir[i] == 1 and rsi[i] < 30 and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI overbought + volume + trending
            elif kama_dir[i] == -1 and rsi[i] > 70 and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA turns down OR RSI overbought
            if kama_dir[i] == -1 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA turns up OR RSI oversold
            if kama_dir[i] == 1 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0