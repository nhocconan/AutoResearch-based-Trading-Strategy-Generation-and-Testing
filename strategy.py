#!/usr/bin/env python3
# 1d_KAMA_Trend_1wRSI_CloseFilter
# Hypothesis: Uses 1d KAMA for trend direction with 1w RSI as a momentum filter and price close relative to KAMA for entry.
# KAMA adapts to market noise, reducing whipsaws in ranging markets. 1w RSI > 50 confirms bullish momentum on weekly scale,
# RSI < 50 confirms bearish momentum. Enter long when price > KAMA and weekly RSI > 50, short when price < KAMA and weekly RSI < 50.
# Exit when price crosses back across KAMA or weekly RSI flips. Designed for 1d timeframe to target 30-100 total trades over 4 years.
# Position size 0.25 for balanced risk management.

name = "1d_KAMA_Trend_1wRSI_CloseFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1d KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    for i in range(1, len(volatility)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1]) - (np.abs(close[i-10] - close[i-11]) if i >= 11 else 0)
    er = change / volatility
    er = np.where(np.isnan(er) | (volatility == 0), 0, er)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate weekly RSI (14-period)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1w, np.nan)
    avg_loss = np.full_like(close_1w, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Align weekly RSI to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Warmup for KAMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(kama[i]) or np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price > KAMA and weekly RSI > 50
            if close[i] > kama[i] and rsi_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short entry: price < KAMA and weekly RSI < 50
            elif close[i] < kama[i] and rsi_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or weekly RSI drops below 50
            if close[i] < kama[i] or rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or weekly RSI rises above 50
            if close[i] > kama[i] or rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals