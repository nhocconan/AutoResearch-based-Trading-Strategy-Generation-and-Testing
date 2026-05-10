#/usr/bin/env python3
"""
4h_RSI_Extreme_Trend_Filter
Hypothesis: Buy when RSI(14) < 30 (oversold) in a bullish trend (price > 200 EMA) and sell when RSI(14) > 70 (overbought) in a bearish trend (price < 200 EMA). Uses 1d EMA200 for trend filter to ensure alignment with long-term direction. RSI extremes in strong trends often precede mean-reversion moves. Works in both bull and bear markets by only taking trades in the direction of the 1d trend, avoiding counter-trend whipsaws. Target: ~25 trades/year (100 total) to minimize fee drag.
"""

name = "4h_RSI_Extreme_Trend_Filter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend filter
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema200_1d[i-1]
    
    # RSI(14) on 4h data
    rsi_period = 14
    rsi = np.full(n, np.nan)
    if n >= rsi_period:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        for i in range(rsi_period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
    
    # Align 1d EMA200 to 4h
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, rsi_period)  # Wait for EMA200 and RSI
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend and RSI conditions
        is_uptrend = close[i] > ema200_1d_aligned[i]
        is_downtrend = close[i] < ema200_1d_aligned[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # Long: RSI oversold in uptrend
            if rsi_oversold and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in downtrend
            elif rsi_overbought and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral or trend breaks
            if rsi[i] >= 50 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral or trend breaks
            if rsi[i] <= 50 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals