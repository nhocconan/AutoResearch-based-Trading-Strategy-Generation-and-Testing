#!/usr/bin/env python3
# 1d_KAMA_Trend_Filter_With_RSI_Pullback_v1
# Hypothesis: Uses KAMA trend direction on 1d timeframe with RSI(14) pullback on 1d for entry.
# Trend filter avoids counter-trend trades; RSI pullback provides entry during retracements.
# Works in bull markets (trend following) and bear markets (avoids longs in downtrend, takes shorts on pullbacks).
# Position size 0.25 for balanced risk; targets ~15-25 trades/year to minimize fee drag.

name = "1d_KAMA_Trend_Filter_With_RSI_Pullback_v1"
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
    
    # Get 1d data for KAMA trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - trend filter
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # 10-period volatility sum
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (already aligned, but keep for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    # Subsequent averages
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan  # First 14 values undefined
    
    # Align RSI to 1d timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from KAMA
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        if position == 0:
            # Long entry: price in uptrend and RSI pulling back from overbought (< 40)
            if uptrend and rsi_aligned[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short entry: price in downtrend and RSI pulling back from oversold (> 60)
            elif downtrend and rsi_aligned[i] > 60:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend turns down or RSI becomes overbought (> 70)
            if not uptrend or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend turns up or RSI becomes oversold (< 30)
            if not downtrend or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals