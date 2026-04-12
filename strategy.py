#!/usr/bin/env python3
"""
4h_1d_kama_rsi_chop_v2
Uses KAMA direction on 4h with RSI filter and daily chop regime filter.
Long when KAMA rises, RSI < 70, and chop > 61.8 (ranging market favors mean reversion).
Short when KAMA falls, RSI > 30, and chop > 61.8.
Exit on opposite KAMA signal or chop < 38.2 (trending market).
Designed for low trade frequency with volatility regime adaptation.
"""

name = "4h_1d_kama_rsi_chop_v2"
timeframe = "4h"
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
    
    # KAMA on 4h
    kama_length = 10
    fast_ema = 2
    slow_ema = 30
    
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI on 4h
    rsi_length = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_length, min_periods=rsi_length).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_length, min_periods=rsi_length).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get daily data for chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Chop on daily
    chop_length = 14
    atr = np.zeros(len(close_1d))
    tr = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i == 0:
            tr[i] = high_1d[i] - low_1d[i]
        else:
            tr[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]),
                       abs(low_1d[i] - close_1d[i-1]))
        atr[i] = tr[i] if i == 0 else (atr[i-1] * (chop_length-1) + tr[i]) / chop_length
    
    highest_high = pd.Series(high_1d).rolling(window=chop_length, min_periods=chop_length).max().values
    lowest_low = pd.Series(low_1d).rolling(window=chop_length, min_periods=chop_length).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / (np.sum(tr) / chop_length * chop_length)) / np.log10(chop_length)
    
    # Align to 4h
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long: KAMA rising, RSI not overbought, choppy market
        if (kama_aligned[i] > kama_aligned[i-1] and 
            rsi_aligned[i] < 70 and 
            chop_aligned[i] > 61.8 and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short: KAMA falling, RSI not oversold, choppy market
        elif (kama_aligned[i] < kama_aligned[i-1] and 
              rsi_aligned[i] > 30 and 
              chop_aligned[i] > 61.8 and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: opposite KAMA signal or trending market
        elif position == 1 and (kama_aligned[i] < kama_aligned[i-1] or chop_aligned[i] < 38.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (kama_aligned[i] > kama_aligned[i-1] or chop_aligned[i] < 38.2):
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals