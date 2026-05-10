#!/usr/bin/env python3
"""
1d_WeeklyTrend_KAMA_Exit
Hypothesis: Go long when weekly close > weekly EMA34 and daily KAMA is rising, short when weekly close < weekly EMA34 and daily KAMA is falling.
Exit when KAMA changes direction. Uses weekly trend filter to avoid counter-trend trades.
Designed for 1d timeframe to target 10-25 trades/year.
Weekly EMA provides strong trend filter; KAMA adapts to volatility for smooth entries/exits.
Works in bull/bear markets by following higher timeframe trend.
"""

name = "1d_WeeklyTrend_KAMA_Exit"
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
    
    # Calculate weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_34_1w[i-1]
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily KAMA (adaptive moving average)
    # Using Efficiency Ratio (ER) method
    kama = np.full(n, np.nan)
    if n >= 30:
        # Calculate change and volatility
        change = np.abs(np.diff(close, n=10))  # 10-period change
        volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
        
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        
        # Initialize KAMA
        kama[29] = np.mean(close[:30])
        for i in range(30, n):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup for KAMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly uptrend and KAMA rising
            if close[i] > ema_34_1w_aligned[i] and kama[i] > kama[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend and KAMA falling
            elif close[i] < ema_34_1w_aligned[i] and kama[i] < kama[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA turns down (end of upward move)
            if kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA turns up (end of downward move)
            if kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals