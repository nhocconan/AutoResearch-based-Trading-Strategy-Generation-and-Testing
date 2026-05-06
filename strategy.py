#!/usr/bin/env python3
# 1d_1wKAMA_Trend_Filtered_By_RSI
# Uses weekly KAMA for trend direction and daily RSI for overbought/oversold entries.
# Long when weekly trend is up and daily RSI < 30 (oversold).
# Short when weekly trend is down and daily RSI > 70 (overbought).
# Designed for 1d timeframe to capture swing reversals in both bull and bear markets.
# Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing.

name = "1d_1wKAMA_Trend_Filtered_By_RSI"
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
    
    # Get weekly data for KAMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA (adaptive moving average)
    close_1w = df_1w['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility properly: sum of absolute changes over window
    volatility = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i == 0:
            volatility[i] = 0
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_1w[max(0, i-9):i+1])))
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Daily RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Average gain and loss
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(len(close)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i] if i == 0 else np.mean(gain[1:i+1])
                avg_loss[i] = loss[i] if i == 0 else np.mean(loss[1:i+1])
            else:
                avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else gain[i]
                avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # warmup for weekly KAMA and daily RSI
        # Skip if any critical value is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend (price above KAMA) and daily RSI oversold (<30)
            if close[i] > kama_aligned[i] and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend (price below KAMA) and daily RSI overbought (>70)
            elif close[i] < kama_aligned[i] and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly trend turns down OR RSI overbought (>70)
            if close[i] < kama_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly trend turns up OR RSI oversold (<30)
            if close[i] > kama_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals