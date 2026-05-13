#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies trend direction, RSI measures momentum strength, and Choppiness Index filters range-bound markets. 
Long when KAMA slope up, RSI > 50, and CHOP > 61.8 (range). Short when KAMA slope down, RSI < 50, and CHOP > 61.8.
Uses 12h EMA50 trend filter for higher timeframe bias. Works in both bull and bear markets by avoiding whipsaws in chop.
Target: 20-40 trades/year per symbol.
"""

name = "4h_KAMA_Direction_RSI_ChopFilter"
timeframe = "4h"
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
    
    # KAMA: Kaufman Adaptive Moving Average
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, np.abs(change) / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr.sum() / (highest_high - lowest_low)) / np.log10(14))
    chop = np.where((highest_high - lowest_low) != 0, 100 * np.log10((atr.sum() / (highest_high - lowest_low)) / np.log10(14)), 50)
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        kama_slope = kama[i] - kama[i-1]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        uptrend_htf = uptrend_12h_aligned[i]
        downtrend_htf = downtrend_12h_aligned[i]
        
        if position == 0:
            # LONG: KAMA slope up, RSI > 50, CHOP > 61.8 (range), 12h uptrend
            if kama_slope > 0 and rsi_val > 50 and chop_val > 61.8 and uptrend_htf:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA slope down, RSI < 50, CHOP > 61.8 (range), 12h downtrend
            elif kama_slope < 0 and rsi_val < 50 and chop_val > 61.8 and downtrend_htf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA slope down or RSI <= 50
            if kama_slope <= 0 or rsi_val <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA slope up or RSI >= 50
            if kama_slope >= 0 or rsi_val >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals