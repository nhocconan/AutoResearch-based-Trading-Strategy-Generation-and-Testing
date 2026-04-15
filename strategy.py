#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA + RSI + Chop Filter
# Uses KAMA for adaptive trend detection, RSI for momentum confirmation,
# and Choppiness Index to filter range-bound markets. KAMA adapts to volatility,
# making it robust in both bull and bear markets. RSI avoids overextended entries.
# Target: 80-180 total trades over 4 years (20-45/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Load 1d data for Chop and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA on 4h (ER=10, fast=2, slow=30)
    change_4h = np.abs(np.diff(close_4h, k=10))
    volatility_4h = np.sum(np.abs(np.diff(close_4h, k=1)), axis=0)
    er_4h = np.where(volatility_4h != 0, change_4h / volatility_4h, 0)
    sc_4h = (er_4h * (2/(2) - 2/(30)) + 2/(30)) ** 2
    kama_4h = np.full_like(close_4h, np.nan)
    kama_4h[9] = close_4h[9]  # Seed
    for i in range(10, len(close_4h)):
        kama_4h[i] = kama_4h[i-1] + sc_4h[i] * (close_4h[i] - kama_4h[i-1])
    
    # Calculate RSI (14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14) on 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    
    # Align indicators to 4h timeframe
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(100, n):
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            continue
        
        # Long: price > KAMA, RSI > 50 and < 70, chop < 61.8 (trending)
        if (close[i] > kama_4h_aligned[i] and
            50 < rsi_1d_aligned[i] < 70 and
            chop_aligned[i] < 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: price < KAMA, RSI < 50 and > 30, chop < 61.8 (trending)
        elif (close[i] < kama_4h_aligned[i] and
              30 < rsi_1d_aligned[i] < 50 and
              chop_aligned[i] < 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or chop > 61.8 (ranging)
        elif position == 1 and (close[i] < kama_4h_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama_4h_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_KAMA_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0