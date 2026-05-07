#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman's Adaptive Moving Average (KAMA) to identify trend direction,
combined with RSI for overbought/oversold conditions and Choppiness Index to avoid choppy markets.
Long when KAMA upward, RSI < 50, and CHOP > 61.8 (range). Short when KAMA downward, RSI > 50, and CHOP > 61.8.
This avoids whipsaws in chop while capturing trend moves. Weekly trend filter ensures alignment with higher timeframe.
Works in both bull and bear markets by requiring alignment with weekly trend and using range-bound signals.
"""
name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === KAMA (10,2,30) ===
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0
    # Sum of absolute daily changes
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(abs_change).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14) ===
    atr = np.maximum(high - low,
                     np.maximum(np.abs(high - np.roll(close, 1)),
                                np.abs(low - np.roll(close, 1))))
    atr[0] = high[0] - low[0]
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(14)
    # Handle division by zero or invalid
    chop = np.where((hh - ll) != 0, chop, 50)
    
    # === Weekly EMA50 for trend filter ===
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 14)  # KAMA(30), RSI(14), CHOP(14)
    
    for i in range(start_idx, n):
        if position == 0:
            # Long: KAMA upward (close > kama), RSI < 50, CHOP > 61.8 (range), price above weekly EMA50
            if (close[i] > kama[i] and
                rsi[i] < 50 and
                chop[i] > 61.8 and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA downward (close < kama), RSI > 50, CHOP > 61.8 (range), price below weekly EMA50
            elif (close[i] < kama[i] and
                  rsi[i] > 50 and
                  chop[i] > 61.8 and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA downward or RSI > 70 (overbought) or CHOP < 38.2 (trending)
            if (close[i] < kama[i] or
                rsi[i] > 70 or
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA upward or RSI < 30 (oversold) or CHOP < 38.2 (trending)
            if (close[i] > kama[i] or
                rsi[i] < 30 or
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals