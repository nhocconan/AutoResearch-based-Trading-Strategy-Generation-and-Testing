#!/usr/bin/env python3
"""
1d_KAMA_Trend_Signal_RSI_Pullback_1wTrend_Filter
Hypothesis: Use 1d KAMA trend direction as primary filter, enter on RSI pullbacks (RSI<30 for long, RSI>70 for short) only when 1-week trend agrees. Exit on opposite RSI extreme (RSI>70 for long exit, RSI<30 for short exit). Designed for low trade frequency (<15/year) to minimize fee drag while capturing trend continuation moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate 1d KAMA for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Efficiency Ratio for KAMA
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align 1d KAMA to 1d timeframe (no alignment needed as we're on 1d)
    kama_aligned = kama
    
    # Calculate 1-week trend using EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 14-period RSI on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = rsi
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI and KAMA
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        rsi_val = rsi_aligned[i]
        close_1d_val = close_1d[i]
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI<30 (oversold), and 1w EMA50 rising
            if close_1d_val > kama_val and rsi_val < 30 and ema_50_1w_val > ema_50_1w_aligned[i-1]:
                signals[i] = size
                position = 1
            # Short: price below KAMA (downtrend), RSI>70 (overbought), and 1w EMA50 falling
            elif close_1d_val < kama_val and rsi_val > 70 and ema_50_1w_val < ema_50_1w_aligned[i-1]:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI>70 (overbought) or price below KAMA
            if rsi_val > 70 or close_1d_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI<30 (oversold) or price above KAMA
            if rsi_val < 30 or close_1d_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_Signal_RSI_Pullback_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0