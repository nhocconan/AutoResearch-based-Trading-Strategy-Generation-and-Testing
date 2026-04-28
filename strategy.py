#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI14_Band
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction and RSI(14) for overbought/oversold conditions.
Goes long when price > KAMA and RSI < 30 (oversold), short when price < KAMA and RSI > 70 (overbought).
Exits when price crosses back across KAMA.
Designed to work in both bull and bear markets by following adaptive trend while avoiding extremes.
Targets ~25-35 trades/year via KAMA trend filter and RSI extreme conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (ER=10, FAST=2, SLOW=30) for trend
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fast = 2.0
    slow = 30.0
    sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        if not np.isnan(avg_gain[i-1]) and not np.isnan(avg_loss[i-1]):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        else:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Wait for RSI to stabilize
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = close[i] > kama[i] and rsi[i] < 30
        short_entry = close[i] < kama[i] and rsi[i] > 70
        
        # Exit when price crosses KAMA
        long_exit = close[i] < kama[i]
        short_exit = close[i] > kama[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_RSI14_Band"
timeframe = "4h"
leverage = 1.0