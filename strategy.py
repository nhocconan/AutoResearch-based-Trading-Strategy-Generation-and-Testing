#!/usr/bin/env python3
"""
12h KAMA + RSI + Chop Regime.
Long when KAMA rising + RSI > 50 + Chop < 61.8 (trending).
Short when KAMA falling + RSI < 50 + Chop < 61.8 (trending).
Exit when Chop > 61.8 (range) or RSI crosses 50 opposite.
Designed for low frequency (12-37 trades/year) with regime filter to avoid whipsaws.
"""

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
    
    # KAMA (adaptive moving average)
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    
    # Calculate ER (Efficiency Ratio)
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(9, n):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Calculate SC (Smoothing Constant)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[9] = close[9]  # Start with close
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    rsi_period = 14
    delta = np.diff(close)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    # Calculate average gain/loss
    avg_up = np.zeros(n)
    avg_down = np.zeros(n)
    avg_up[:] = np.nan
    avg_down[:] = np.nan
    
    if n >= rsi_period:
        avg_up[rsi_period-1] = np.mean(up[:rsi_period])
        avg_down[rsi_period-1] = np.mean(down[:rsi_period])
        for i in range(rsi_period, n):
            avg_up[i] = (avg_up[i-1] * (rsi_period-1) + up[i-1]) / rsi_period
            avg_down[i] = (avg_down[i-1] * (rsi_period-1) + down[i-1]) / rsi_period
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(rsi_period, n):
        if avg_down[i] != 0:
            rs = avg_up[i] / avg_down[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # Choppy Index (14-period)
    chop_period = 14
    atr = np.zeros(n)
    atr[:] = np.nan
    
    # True Range
    tr = np.zeros(n)
    tr[:] = np.nan
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # ATR
    if n >= chop_period:
        atr[chop_period-1] = np.mean(tr[1:chop_period])
        for i in range(chop_period, n):
            atr[i] = (atr[i-1] * (chop_period-1) + tr[i]) / chop_period
    
    # Chop calculation
    chop = np.zeros(n)
    chop[:] = np.nan
    for i in range(chop_period, n):
        atr_sum = np.sum(atr[i-chop_period+1:i+1])
        max_high = np.max(high[i-chop_period+1:i+1])
        min_low = np.min(low[i-chop_period+1:i+1])
        if max_high != min_low:
            chop[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(chop_period)
        else:
            chop[i] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA(10), RSI(14), Chop(14)
    start_idx = max(10, 14, 14) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        kama_now = kama[i]
        kama_prev = kama[i-1]
        rsi_now = rsi[i]
        chop_now = chop[i]
        
        # Regime filter: trending market (Chop < 61.8)
        trending = chop_now < 61.8
        
        # KAMA direction
        kama_rising = kama_now > kama_prev
        kama_falling = kama_now < kama_prev
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + trending
            if kama_rising and rsi_now > 50 and trending:
                signals[i] = size
                position = 1
            # Short: KAMA falling + RSI < 50 + trending
            elif kama_falling and rsi_now < 50 and trending:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Chop > 61.8 (range) or RSI < 50
            if chop_now > 61.8 or rsi_now < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Chop > 61.8 (range) or RSI > 50
            if chop_now > 61.8 or rsi_now > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_RSI_Chop"
timeframe = "12h"
leverage = 1.0