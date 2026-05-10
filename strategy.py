#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_WeeklyTrend
Hypothesis: Use daily KAMA for trend direction, RSI for momentum filter, and weekly trend for regime.
KAMA adapts to market conditions, reducing whipsaw in sideways markets. RSI avoids overbought/oversold extremes.
Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
Works in both bull and bear markets by following weekly trend and using adaptive trend filtering.
Targets 10-20 trades/year per symbol with discrete position sizing to minimize fee drag.
"""

name = "1d_KAMA_Trend_RSI_WeeklyTrend"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA(10, 2, 30) for trend
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility[9:]])  # align lengths
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = np.mean(close[:10])  # seed
    for i in range(10, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) for momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    # Wilder smoothing
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate weekly trend using EMA(34) on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_34_1w[i-1]
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure KAMA, RSI, and weekly EMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend), RSI not overbought, and weekly trend up
            if close[i] > kama[i] and rsi[i] < 70 and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend), RSI not oversold, and weekly trend down
            elif close[i] < kama[i] and rsi[i] > 30 and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below KAMA or RSI overbought
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above KAMA or RSI oversold
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals