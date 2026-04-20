#!/usr/bin/env python3
# 4h_1d_KAMA_RSI_ChopFilter_v2
# Hypothesis: Use 1d KAMA for trend direction, 4h RSI for entry timing, and 4h Choppiness Index to filter ranging markets.
# In trending markets (Chop < 38.2), enter long when price > 1d KAMA and RSI < 40; enter short when price < 1d KAMA and RSI > 60.
# Exit on opposite signal or when Chop > 61.8 (strong ranging). Designed to avoid whipsaws in chop and capture trends in both bull/bear markets.
# Target: 20-40 trades per year per symbol to stay within fee limits.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d KAMA ( Kaufman Adaptive Moving Average ) ===
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    er = np.where(vol != 0, change / vol, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === 4h RSI(14) ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h Choppiness Index (14) ===
    high = prices['high'].values
    low = prices['low'].values
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) != 0, chop, 50)  # avoid division by zero
    
    # Align 1d KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Chop/RSI warmup
        # Get values
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long in trending market (Chop < 38.2) when price > KAMA and RSI < 40 (oversold)
            if chop_val < 38.2 and close_val > kama_val and rsi_val < 40:
                signals[i] = 0.25
                position = 1
            # Enter short in trending market (Chop < 38.2) when price < KAMA and RSI > 60 (overbought)
            elif chop_val < 38.2 and close_val < kama_val and rsi_val > 60:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: reverse signal or market becomes ranging (Chop > 61.8)
            if (chop_val > 61.8) or (close_val < kama_val) or (rsi_val > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: reverse signal or market becomes ranging (Chop > 61.8)
            if (chop_val > 61.8) or (close_val > kama_val) or (rsi_val < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals