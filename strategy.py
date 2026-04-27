#!/usr/bin/env python3
"""
Hypothesis: 1-day mean reversion using RSI extremes with weekly trend filter.
In both bull and bear markets, price tends to revert to mean when RSI reaches extremes,
but only in the direction of the weekly trend to avoid counter-trend losses.
Uses weekly EMA40 as trend filter and RSI(14) for entry/exit.
Target: 10-25 trades per year with low turnover to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA40 for trend filter
    ema_period = 40
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate daily RSI(14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # First average
    if n >= rsi_period:
        avg_gain[rsi_period - 1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period - 1] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    for i in range(rsi_period - 1, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI and weekly EMA
    start_idx = max(rsi_period, ema_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(rsi[i]) or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_trend_up = price > ema_1w_aligned[i]
        weekly_trend_down = price < ema_1w_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) in weekly uptrend
            if rsi[i] < 30 and weekly_trend_up:
                signals[i] = size
                position = 1
            # Short: RSI overbought (>70) in weekly downtrend
            elif rsi[i] > 70 and weekly_trend_down:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (>50)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (<50)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RSI14_MeanReversion_1wEMA40_Trend"
timeframe = "1d"
leverage = 1.0