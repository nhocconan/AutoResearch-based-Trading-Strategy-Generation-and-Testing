#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_ChopFilter
# Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) for trend direction, RSI for overbought/oversold signals, and Choppiness Index for regime filtering. 
# Enter long when KAMA slopes up, RSI < 30, and market is choppy (mean reversion regime). 
# Enter short when KAMA slopes down, RSI > 70, and market is choppy. 
# This strategy aims to capture mean reversion in choppy markets while avoiding strong trends, suitable for both bull and bear markets.

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
    volume = prices['volume'].values

    # Get weekly data for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate KAMA (Kaufman Adaptive Moving Average) for daily trend
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close(t) - close(t-10)|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close(t) - close(t-1)| over 10 periods
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    # Align KAMA is not needed as it's already on daily timeframe

    # Calculate RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for the first element
    rsi = np.concatenate([[np.nan], rsi])

    # Calculate Choppiness Index (14)
    # True Range
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - low)
    tr3 = np.abs(np.roll(low, 1) - high)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    cpi = 100 * np.log10(sum_tr / (max_hh - min_ll)) / np.log10(14)
    # Handle division by zero or invalid cases
    cpi = np.where((max_hh - min_ll) == 0, np.nan, cpi)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(cpi[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising, RSI oversold, choppy market (mean reversion regime)
            if kama[i] > kama[i-1] and rsi[i] < 30 and cpi[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI overbought, choppy market (mean reversion regime)
            elif kama[i] < kama[i-1] and rsi[i] > 70 and cpi[i] > 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or RSI overbought or market trends
            if kama[i] < kama[i-1] or rsi[i] > 70 or cpi[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or RSI oversold or market trends
            if kama[i] > kama[i-1] or rsi[i] < 30 or cpi[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals