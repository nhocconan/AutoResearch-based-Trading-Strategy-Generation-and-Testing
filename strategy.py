#!/usr/bin/env python3
name = "4h_KAMA_RSI_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA on close (ER=10)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    sc = (er * 0.2 + (1 - er) * 0.067) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) on close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily EMA34 trend
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Signals
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(ema34_aligned[i]) or np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            if close[i] > kama_aligned[i] and ema34_aligned[i] and rsi_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            elif close[i] < kama_aligned[i] and not ema34_aligned[i] and rsi_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if close[i] < kama_aligned[i] or not ema34_aligned[i] or rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] > kama_aligned[i] or ema34_aligned[i] or rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals