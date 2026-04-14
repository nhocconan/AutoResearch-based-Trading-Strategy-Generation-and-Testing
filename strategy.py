#!/usr/bin/env python3
"""
12h KAMA + RSI + Chop Filter (v1)
- KAMA(10) direction from 12h close (trend filter)
- RSI(14) from 12h (momentum filter: long if RSI > 50, short if RSI < 50)
- Chop(14) from 1d (regime filter: trade only if Chop > 61.8 [range])
- Position size: 0.25
- Target: 15-25 trades/year per symbol (60-100 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA(10) from 12h close
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    price_change = np.abs(np.diff(close_12h, n=10))  # 10-period change
    abs_sum = np.sum(np.abs(np.diff(close_12h)), axis=0)  # placeholder
    
    # Recompute properly
    change_10 = np.abs(np.diff(close_12h, n=10))
    sum_abs = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        sum_abs[i] = sum_abs[i-1] + np.abs(close_12h[i] - close_12h[i-1])
    er = np.where(sum_abs > 0, change_10 / sum_abs, 0)
    
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # seed
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Calculate RSI(14) from 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Chop(14) from 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Max/Min over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = np.where((max_high - min_low) > 0, 
                    100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14), 
                    50)
    # Fix chop calculation
    chop = np.full_like(close_1d, 50.0)
    for i in range(13, len(close_1d)):
        tr_sum = np.nansum(tr[i-13:i+1])
        range_14 = max_high[i] - min_low[i]
        if range_14 > 0:
            chop[i] = 100 * np.log10(tr_sum / range_14) / np.log10(14)
    
    # Align indicators to 12h
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # Align chop to 12h (need to align from 1d to 12h)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            continue
        
        # Chop filter: only trade in ranging markets (Chop > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        if not chop_filter:
            continue
        
        if position == 0:  # No position - look for entries
            # Long: price > KAMA and RSI > 50
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50:
                position = 1
                signals[i] = position_size
            # Short: price < KAMA and RSI < 50
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50:
                position = -1
                signals[i] = -position_size
        elif position == 1:  # Long position - exit when price < KAMA or RSI < 50
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price > KAMA or RSI > 50
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_RSI_Chop_v1"
timeframe = "12h"
leverage = 1.0