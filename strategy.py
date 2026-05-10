#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Signal
Hypothesis: Daily KAMA trend filter with RSI momentum for entries.
Uses weekly trend as higher timeframe filter to avoid counter-trend trades.
KAMA adapts to market noise, reducing false signals in ranging markets.
RSI provides momentum confirmation for entries.
Target: 15-25 trades/year (60-100 total over 4 years).
Works in bull/bear by following weekly trend direction.
"""

name = "1d_KAMA_Trend_RSI_Signal"
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
    
    # Weekly trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    if len(close_1w) >= 50:
        sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    else:
        sma_50_1w = np.full(len(close_1w), np.nan)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Daily KAMA trend (adaptive moving average)
    # Efficiency Ratio calculation
    price_change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros_like(close)
    er[10:] = price_change[10:] / np.maximum(volatility[10:], 1e-10)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI for momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Wilder's smoothing
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # KAMA needs some history, RSI needs 14
    
    for i in range(start_idx, n):
        if np.isnan(sma_50_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend alignment: price relative to weekly SMA50
        weekly_bullish = close[i] > sma_50_1w_aligned[i]
        weekly_bearish = close[i] < sma_50_1w_aligned[i]
        
        # KAMA trend direction
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        if position == 0:
            # Long: Weekly bullish, KAMA rising, RSI > 50 (momentum)
            if weekly_bullish and kama_rising and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: Weekly bearish, KAMA falling, RSI < 50 (momentum)
            elif weekly_bearish and kama_falling and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Weekly bearish OR KAMA falling
            if not weekly_bullish or not kama_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Weekly bullish OR KAMA rising
            if not weekly_bearish or not kama_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals