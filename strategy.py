#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Pullback
Hypothesis: KAMA trend direction with RSI pullback entries on 4h timeframe.
Uses KAMA to establish trend (adaptive to noise) and RSI for mean-reversion entries.
Works in both bull and bear markets by following trend and buying dips/selling rallies.
Target: 20-30 trades/year to minimize fee drag.
"""

name = "4h_KAMA_Trend_RSI_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Use rolling sum for volatility
    volatility_rolling = pd.Series(np.abs(np.diff(close_1d, prepend=close_1d[0]))).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility_rolling > 0, change / volatility_rolling, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Get price, RSI, volume
    close = prices['close'].values
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    volume = prices['volume'].values
    # Volume filter: current volume > 1.5x 20-period SMA
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_sma20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14) and volume SMA (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend) and RSI < 40 (pullback) with volume
            if close[i] > kama_1d_aligned[i] and rsi[i] < 40 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) and RSI > 60 (pullback) with volume
            elif close[i] < kama_1d_aligned[i] and rsi[i] > 60 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 60 (overbought) or trend change
            if rsi[i] > 60 or close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 40 (oversold) or trend change
            if rsi[i] < 40 or close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals