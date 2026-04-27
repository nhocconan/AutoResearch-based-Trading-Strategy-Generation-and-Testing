#!/usr/bin/env python3
"""
Hypothesis: 4h KAMA (Adaptive Moving Average) with RSI filter and 12h trend filter.
- KAMA adapts to market volatility, reducing whipsaws in choppy markets
- RSI(14) < 30 for long, > 70 for short to capture mean reversions
- 12h EMA(50) filter ensures alignment with medium-term trend
- Exit on opposite KAMA crossover or RSI returning to neutral (40-60)
- Target: 20-30 trades/year to avoid fee drag
- Uses discrete position sizing (0.25) to minimize churn
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h data
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_12h[49] = np.mean(close_12h[:50])  # Simple average for first value
        for i in range(50, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 / (50 + 1)) + (ema_12h[i-1] * (49 / (50 + 1)))
    
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER (Efficiency Ratio) and SC (Smoothing Constant)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will compute properly
    
    # Proper ER calculation
    er = np.full(n, np.nan)
    for i in range(10, n):  # 10-period ER
        if i >= 10:
            net_change = np.abs(close[i] - close[i-10])
            total_change = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if total_change > 0:
                er[i] = net_change / total_change
            else:
                er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        else:
            sc[i] = 0
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # First average (simple)
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        
        # Wilder's smoothing
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    for i in range(14, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100 if avg_gain[i] > 0 else 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price > KAMA + RSI < 30 + price > 12h EMA
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < KAMA + RSI > 70 + price < 12h EMA
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  close[i] < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < KAMA OR RSI > 50 (mean reversion) OR price < 12h EMA (trend change)
            if (close[i] < kama[i] or 
                rsi[i] > 50 or 
                close[i] < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > KAMA OR RSI < 50 (mean reversion) OR price > 12h EMA (trend change)
            if (close[i] > kama[i] or 
                rsi[i] < 50 or 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI30_70_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0