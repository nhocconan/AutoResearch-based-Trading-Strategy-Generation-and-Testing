#!/usr/bin/env python3
name = "1d_KAMA_Direction_RSI_Chop_Filter_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== KAMA (Kaufman Adaptive Moving Average) =====
    # Efficiency Ratio (ER) = |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.diff(close, n=10, prepend=close[:10]))
    er = direction / (np.sum(change.reshape(-1, 1) * np.tril(np.ones((10, 10))), axis=1) + 1e-10)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ===== RSI(14) =====
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ===== Choppiness Index (14) =====
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr.reshape(-1, 1) * np.tril(np.ones((14, 14))), axis=1) / 
                          (np.log10(max_high - min_low) * 14)) if np.any(max_high > min_low) else 50
    chop = np.where(max_high == min_low, 50, chop)
    
    # ===== Signals =====
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + Chop < 61.8 (trending)
            if kama[i] > kama[i-1] and rsi[i] > 50 and chop[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + Chop < 61.8 (trending)
            elif kama[i] < kama[i-1] and rsi[i] < 50 and chop[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down OR RSI < 40 OR Chop > 61.8 (choppy)
            if kama[i] < kama[i-1] or rsi[i] < 40 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up OR RSI > 60 OR Chop > 61.8 (choppy)
            if kama[i] > kama[i-1] or rsi[i] > 60 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals