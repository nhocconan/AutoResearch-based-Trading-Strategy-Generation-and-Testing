#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter
Hypothesis: KAMA adapts to market noise, capturing trends while avoiding whipsaws in ranging markets.
Combined with RSI for momentum confirmation and Choppiness Index to filter ranging conditions,
this strategy aims to capture sustained trends in both bull and bear markets with low trade frequency.
Designed for ~15-25 trades/year to minimize fee decay.
"""
name = "1d_KAMA_RSI_Chop"
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
    
    # === KAMA (10) ===
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np, 'sum') else np.abs(np.diff(close)).sum()
    # Manual ER calculation to avoid deprecation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        price_change = np.abs(close[i] - close[i-10])
        price_volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
        er[i] = price_change / price_volatility if price_volatility != 0 else 0
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[9] = close[9]
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14) ===
    atr1 = []
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    for i in range(len(tr)):
        if i < 14:
            atr1.append(np.nan)
        else:
            atr1.append(np.sum(tr[i-13:i+1]) / 14)
    atr1 = np.array(atr1)
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1 * 14 / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA + RSI > 50 + Chop < 61.8 (trending)
            if (close[i] > kama[i] and 
                rsi[i] > 50 and
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + RSI < 50 + Chop < 61.8 (trending)
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below KAMA OR RSI < 40
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR RSI > 60
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals