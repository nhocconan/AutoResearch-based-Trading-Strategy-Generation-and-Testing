#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_ChopFilter
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction on 4h timeframe.
# Enters long when price is above KAMA and RSI > 50 with chop filter (Choppiness Index > 61.8 = ranging).
# Enters short when price is below KAMA and RSI < 50 with chop filter.
# Exits when price crosses KAMA in opposite direction.
# Uses Chop filter to avoid whipsaws in strong trends and only trade in ranging markets.
# Targets 20-40 trades per year on 4h timeframe with position size 0.25.

name = "4h_KAMA_Direction_RSI_ChopFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (ER=10, FAST=2, SLOW=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Fix: volatility should be rolling sum of absolute changes
    volatility = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14)
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(np.sum(atr, axis=1) / (max_high - min_low)) / np.log10(14), 
                    50)
    # Fix chop calculation: sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) != 0, chop, 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 30)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: only trade when market is ranging (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        if position == 0:
            # Long entry: price above KAMA, RSI > 50, and chop filter
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA, RSI < 50, and chop filter
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals