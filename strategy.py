#!/usr/bin/env python3
"""
12h_KAMA_Trend_RangeFilter
Hypothesis: KAMA adapts to market noise, staying close in trends and whipsawing in ranges.
Go long when price > KAMA and range expansion (ATR rising), short when price < KAMA and ATR rising.
Use 1-day trend filter to avoid counter-trend trades. Designed for low turnover in both bull and bear markets.
Target: 15-30 trades/year on 12h timeframe.
"""

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
    
    # KAMA parameters
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, n=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    
    er = np.zeros_like(change)
    for i in range(len(change)):
        if vol[i] > 0:
            er[i] = change[i] / vol[i]
        else:
            er[i] = 0
    
    # Pad ER to match close length
    er_padded = np.concatenate([np.full(10, np.nan), er])
    
    sc = (er_padded * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = np.mean(close[0:10])  # Seed with 10-period SMA
    
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ATR for volatility and range expansion
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR rising: current > 1-period ago
    atr_rising = np.zeros_like(atr, dtype=bool)
    for i in range(1, len(atr)):
        if not np.isnan(atr[i]) and not np.isnan(atr[i-1]):
            atr_rising[i] = atr[i] > atr[i-1]
    
    # 1-day EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full_like(close_1d, np.nan)
    k = 2 / (34 + 1)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema34_1d[i] = np.mean(close_1d[0:35])
        else:
            ema34_1d[i] = close_1d[i] * k + ema34_1d[i-1] * (1 - k)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: above average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(atr_rising[i]) or np.isnan(vol_ok[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA, ATR rising (breakout), 1-day uptrend, volume
            if (close[i] > kama[i] and atr_rising[i] and 
                close[i] > ema34_1d_aligned[i] and vol_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, ATR rising (breakdown), 1-day downtrend, volume
            elif (close[i] < kama[i] and atr_rising[i] and 
                  close[i] < ema34_1d_aligned[i] and vol_ok[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA or 1-day trend turns down
            if (close[i] < kama[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA or 1-day trend turns up
            if (close[i] > kama[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_RangeFilter"
timeframe = "12h"
leverage = 1.0