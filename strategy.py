#!/usr/bin/env python3
# 4h_1d_KAMA_RSI_ChopFilter
# Hypothesis: 4h KAMA direction with RSI pullback and 1d Choppiness index regime filter.
# Long when KAMA up, RSI<40, CHOP>61.8 (range); Short when KAMA down, RSI>60, CHOP>61.8.
# Uses 1d Chop to avoid trending markets where mean reversion fails.
# Target: 20-40 trades per year per symbol to avoid fee drag, works in range-bound markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 1d Choppiness Index (14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR14
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log(sum(ATR14)/ (max(high)-min(low))) / log(14)
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log(sum_atr / (max_high - min_low)) / np.log(14)
    
    # === 4h KAMA (10,2,30) ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, 10))  # net change over 10 periods
    vol = np.sum(np.abs(np.diff(close)), axis=0) if False else None  # placeholder
    
    # Efficiency Ratio
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if np.sum(np.abs(np.diff(close[i-9:i+1]))) > 0:
            er[i] = np.abs(close[i] - close[i-10]) / np.sum(np.abs(np.diff(close[i-9:i+1])))
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 4h RSI (14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])  # align
    
    # Align 1d Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Get values
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI<40 (pullback), Chop>61.8 (range)
            if close_val > kama_val and rsi_val < 40 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI>60 (pullback), Chop>61.8 (range)
            elif close_val < kama_val and rsi_val > 60 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI>60 or Chop<38.2 (trending)
            if rsi_val > 60 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI<40 or Chop<38.2 (trending)
            if rsi_val < 40 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals