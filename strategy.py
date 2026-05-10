#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI14_ChopFilter
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction on 4h.
# Goes long when KAMA turns upward and RSI(14) > 50, short when KAMA turns downward and RSI(14) < 50.
# Includes Choppiness Index (CHOP) filter to avoid ranging markets (CHOP > 61.8 = range, avoid trading).
# Designed to work in both bull and bear markets by following adaptive trend.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4h_KAMA_Direction_RSI14_ChopFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    # Proper volatility calculation: sum of absolute changes over ER period
    er_period = 10
    change_abs = np.abs(np.diff(close, prepend=close[0]))
    volatility_sum = np.zeros_like(change_abs)
    for i in range(1, len(volatility_sum)):
        volatility_sum[i] = volatility_sum[i-1] + change_abs[i]
        if i >= er_period:
            volatility_sum[i] -= change_abs[i - er_period]
    er = np.where(volatility_sum > 0, change_abs / volatility_sum, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP) - using 14-period
    atr_list = []
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0, 100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14), 50)
    # Fix: proper CHOP calculation
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) > 0, chop, 50)
    
    # KAMA direction: upward if current > previous, downward if current < previous
    kama_up = kama > np.roll(kama, 1)
    kama_down = kama < np.roll(kama, 1)
    kama_up[0] = False
    kama_down[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # warmup for RSI and CHOP
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: avoid ranging markets (CHOP > 61.8)
        if chop[i] > 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: KAMA upward and RSI > 50
            if kama_up[i] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA downward and RSI < 50
            elif kama_down[i] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA downward or RSI < 50
            if kama_down[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA upward or RSI > 50
            if kama_up[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals