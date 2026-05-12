#!/usr/bin/env python3
"""
1d_KAMA_Plus_RSI_With_Chop_Filter
Hypothesis: KAMA identifies the 1-day trend direction, RSI filters for overbought/oversold conditions within the trend, and Choppiness Index avoids ranging markets. This combination works in both bull and bear markets by following the trend only when momentum aligns and markets are trending (not choppy).
"""

name = "1d_KAMA_Plus_RSI_With_Chop_Filter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data ONCE before loop for Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1-day data
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    
    # Handle the array operations properly
    change_full = np.concatenate([np.full(10, np.nan), change])
    volatility_full = np.concatenate([np.full(10, np.nan), volatility])
    
    er = np.where(volatility_full != 0, change_full / volatility_full, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # RSI (14-period) on 1-day data
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Choppiness Index (14-period) on 1-week data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(np.diff(high_1w, prepend=high_1w[0]))
    tr2 = np.abs(np.diff(low_1w, prepend=low_1w[0]))
    tr3 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        tr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh = np.full_like(high_1w, np.nan)
    ll = np.full_like(low_1w, np.nan)
    for i in range(13, len(high_1w)):
        hh[i] = np.max(high_1w[i-13:i+1])
        ll[i] = np.min(low_1w[i-13:i+1])
    
    # Choppiness Index
    chop = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        if tr_sum[i] > 0 and hh[i] != ll[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after sufficient warmup
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when market is trending (Choppiness < 61.8)
        if chop_1w_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        # LONG: Price above KAMA + RSI not overbought
        if close[i] > kama_1d_aligned[i] and rsi_1d_aligned[i] < 70:
            signals[i] = 0.25
        # SHORT: Price below KAMA + RSI not oversold
        elif close[i] < kama_1d_aligned[i] and rsi_1d_aligned[i] > 30:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals