#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 12h KAMA (Kaufman Adaptive Moving Average) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    abs_change = np.abs(np.diff(close_12h))
    er = np.zeros_like(close_12h)
    er[1:] = change[1:] / (np.abs(np.diff(close_12h, n=10)) + 1e-10)  # 10-period change
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # === 12h RSI(14) ===
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # === 1d Chopiness Index (CHOP) for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 40
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Chop regime: CHOP > 50 = ranging market (mean revert)
        ranging = chop_aligned[i] > 50
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price > KAMA + RSI > 50 + ranging market
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and ranging:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < KAMA + RSI < 50 + ranging market
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and ranging:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when price crosses below KAMA or RSI < 40
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above KAMA or RSI > 60
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_Chop_Range"
timeframe = "12h"
leverage = 1.0