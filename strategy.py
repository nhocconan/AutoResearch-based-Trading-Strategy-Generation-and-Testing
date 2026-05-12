#!/usr/bin/env python3
# 4H_KAMA_DIRECTION_RSI_FILTER
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise—fast in trends, slow in ranges.
# In 4h timeframe, go long when KAMA slope > 0 and RSI < 50 (avoid overbought), short when KAMA slope < 0 and RSI > 50 (avoid oversold).
# Uses 1d EMA50 as trend filter to avoid counter-trend trades. Works in both bull and bear markets by aligning with higher timeframe trend.
# Target: 20-30 trades/year on 4h timeframe.

name = "4H_KAMA_DIRECTION_RSI_FILTER"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h KAMA calculation
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    # KAMA slope (1-bar change)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d EMA50 to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(kama_slope[i]) or 
            np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend + KAMA rising + RSI < 50 (not overbought)
            if (close[i] > ema50_1d_aligned[i] and 
                kama_slope[i] > 0 and 
                rsi[i] < 50):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + KAMA falling + RSI > 50 (not oversold)
            elif (close[i] < ema50_1d_aligned[i] and 
                  kama_slope[i] < 0 and 
                  rsi[i] > 50):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or KAMA flattening
            if (close[i] <= ema50_1d_aligned[i] or 
                kama_slope[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or KAMA flattening
            if (close[i] >= ema50_1d_aligned[i] or 
                kama_slope[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals