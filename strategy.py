#!/usr/bin/env python3
name = "1d_KAMA_RSI_ChopFilter_Trend"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w KAMA for trend direction
    close_1w = df_1w['close'].values
    delta = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    er = np.zeros_like(close_1w)
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(delta.reshape(-1, 10), axis=1)  # 10-period ER
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # k=2, sc=30
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    
    # Align 1w KAMA to daily
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # RSI(14) on daily
    delta_close = np.diff(close, prepend=close[0])
    gain = np.where(delta_close > 0, delta_close, 0)
    loss = np.where(delta_close < 0, -delta_close, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) on daily
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr, axis=1) / (highest - lowest + 1e-10)) / np.log10(14)
    # Fix chop calculation: sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (highest - lowest + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI > 50, chop < 61.8 (trending)
            if close[i] > kama_1w_aligned[i] and rsi[i] > 50 and chop[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI < 50, chop < 61.8 (trending)
            elif close[i] < kama_1w_aligned[i] and rsi[i] < 50 and chop[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA OR RSI < 40
            if close[i] < kama_1w_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA OR RSI > 60
            if close[i] > kama_1w_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals