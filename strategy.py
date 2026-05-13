#!/usr/bin/env python3
name = "1D_KAMA_RSI_Chop_Reversal_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA trend (1d)
    er = np.zeros(n)
    for i in range(2, n):
        price_change = abs(close[i] - close[i-2])
        price_sum = np.sum(np.abs(close[i-1:i+1] - close[i-2:i]))
        er[i] = price_change / price_sum if price_sum != 0 else 0
    
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14 if i >= 1 else tr[i]
    
    chop = np.full(n, 50.0)
    for i in range(13, n):
        sum_atr = np.sum(atr[i-13:i+1])
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        chop[i] = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_up = close[i] > kama[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        chop_high = chop[i] > 61.8  # ranging market
        
        if position == 0:
            # Long in ranging market when oversold
            if chop_high and rsi_oversold and kama_up:
                signals[i] = 0.25
                position = 1
            # Short in ranging market when overbought
            elif chop_high and rsi_overbought and not kama_up:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when overbought or trend changes
            if rsi[i] > 70 or not kama_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when oversold or trend changes
            if rsi[i] < 30 or kama_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals