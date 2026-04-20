#!/usr/bin/env python3
"""
12h_KAMA_1dRSI_1wTrend_Filter
Concept: 12h KAMA trend direction with 1d RSI overbought/oversold and 1w trend filter.
- Long: KAMA rising AND RSI < 30 AND 1w EMA(50) > EMA(200)
- Short: KAMA falling AND RSI > 70 AND 1w EMA(50) < EMA(200)
- Exit: Opposite KAMA direction change
- Position sizing: 0.25
- Target: 50-150 total trades over 4 years
- Works in bull/bear: KAMA adapts to volatility, RSI captures mean reversion in ranges, weekly trend filters counter-trend moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_1dRSI_1wTrend_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 12h: KAMA Calculation ===
    close_12h = df_12h['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)  # 10-period sum of absolute changes
    # Pad the beginning with NaN for alignment
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_diff = np.diff(kama, prepend=kama[0])
    kama_dir = np.where(kama_diff > 0, 1, np.where(kama_diff < 0, -1, 0))
    
    # Align KAMA direction to 12h
    kama_dir_aligned = align_htf_to_ltf(prices, df_12h, kama_dir.astype(float))
    
    # === 1d: RSI(14) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Align RSI to 12h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 1w: EMA Trend Filter (50 and 200) ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Align 1w EMAs to 12h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        kama_dir = kama_dir_aligned[i]
        rsi_val = rsi_aligned[i]
        ema50 = ema_50_1w_aligned[i]
        ema200 = ema_200_1w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_dir) or np.isnan(rsi_val) or 
            np.isnan(ema50) or np.isnan(ema200)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising AND RSI oversold AND weekly uptrend
            if kama_dir == 1 and rsi_val < 30 and ema50 > ema200:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND RSI overbought AND weekly downtrend
            elif kama_dir == -1 and rsi_val > 70 and ema50 < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down
            if kama_dir == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up
            if kama_dir == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals