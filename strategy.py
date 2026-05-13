#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) captures trend direction with minimal lag in trending markets, while RSI filters overbought/oversold conditions and Choppiness Index (CHOP) identifies ranging vs trending regimes. Long when KAMA upward, RSI < 70, and CHOP < 38.2 (trending); Short when KAMA downward, RSI > 30, and CHOP < 38.2. Uses weekly trend filter for higher timeframe bias. Designed to work in both bull and bear markets by avoiding false signals in high-chop regimes.
Target: 15-25 trades/year per symbol.
"""

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Kaufman Adaptive Moving Average (KAMA)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[:, None], axis=1)  # This is incorrect, need rolling sum
    # Correct calculation of volatility as rolling sum of absolute changes
    volatility = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Relative Strength Index (RSI)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (CHOP)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of True Range over period
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max and min close over period
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_close - min_close) != 0, 100 * np.log10(sum_tr / (max_close - min_close)) / np.log10(14), 50)
    
    # Weekly trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        close_val = close[i]
        
        uptrend_htf = uptrend_1w_aligned[i]
        downtrend_htf = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: KAMA upward (price > KAMA), RSI < 70, CHOP < 38.2 (trending), weekly uptrend
            if close_val > kama_val and rsi_val < 70 and chop_val < 38.2 and uptrend_htf:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA downward (price < KAMA), RSI > 30, CHOP < 38.2 (trending), weekly downtrend
            elif close_val < kama_val and rsi_val > 30 and chop_val < 38.2 and downtrend_htf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < KAMA or RSI >= 70 (overbought) or CHOP >= 61.8 (choppy)
            if close_val < kama_val or rsi_val >= 70 or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > KAMA or RSI <= 30 (oversold) or CHOP >= 61.8 (choppy)
            if close_val > kama_val or rsi_val <= 30 or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals