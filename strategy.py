#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1d_Trend_Filter
Hypothesis: Use 4h KAMA (adaptive EMA) to determine primary trend direction, filtered by 1d EMA(34) to ensure alignment with higher timeframe trend. Enter long when 4h price crosses above KAMA and 1d EMA(34) is rising; enter short when price crosses below KAMA and 1d EMA(34) is falling. Exit on opposite cross. Designed for 15-25 trades/year per symbol, works in both bull and bear markets by following the dominant trend.
"""

name = "4h_KAMA_Trend_With_1d_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h price data
    close = prices['close'].values
    
    # Calculate 4h KAMA (adaptive EMA)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after KAMA warmup (index 9) and ensure EMA is valid
    start_idx = 10
    
    for i in range(start_idx, n):
        # Skip if EMA or KAMA is NaN
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend: rising EMA(34) = bullish, falling = bearish
        if i > 0:
            ema_rising = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            ema_falling = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: price crosses above KAMA AND 1d EMA(34) is rising
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and ema_rising:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA AND 1d EMA(34) is falling
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and ema_falling:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals