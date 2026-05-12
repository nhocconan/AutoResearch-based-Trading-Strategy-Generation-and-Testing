#!/usr/bin/env python3
# 12h_KAMA_RSI_Chop_Filter_v2
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction on 12h, confirmed by RSI(14) for momentum and Choppiness Index (CHOP) for regime filtering. Enter long when KAMA slope > 0, RSI > 50, and CHOP > 61.8 (ranging market); short when KAMA slope < 0, RSI < 50, and CHOP > 61.8. Exit when any condition fails. Targets 12-30 trades/year to avoid fee drag and work in ranging markets where mean reversion works.

name = "12h_KAMA_RSI_Chop_Filter_v2"
timeframe = "12h"
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

    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate KAMA (ER=10, FC=2, SC=30) on 1d close
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    abs_change = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if False else None  # placeholder for correct logic
    # Correct ER calculation
    er = np.zeros_like(close_1d, dtype=np.float64)
    for i in range(len(close_1d)):
        if i < 10:
            er[i] = np.nan
        else:
            direction = np.abs(close_1d[i] - close_1d[i-10])
            volatility = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    sc = (er * (2/2 - 1/30) + 1/30) ** 2  # smoothing constant
    kama = np.full_like(close_1d, np.nan, dtype=np.float64)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])

    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Calculate Choppiness Index (CHOP) on 1d
    atr = np.zeros_like(close_1d)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((max_high - min_low) != 0, chop, 50)

    # Align 1d indicators to 12h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)

    # KAMA slope (1-period change)
    kama_slope = np.diff(kama_aligned, prepend=kama_aligned[0])

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_slope[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA slope up, RSI > 50, CHOP > 61.8 (ranging market)
            if (kama_slope[i] > 0 and 
                rsi_aligned[i] > 50 and
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA slope down, RSI < 50, CHOP > 61.8 (ranging market)
            elif (kama_slope[i] < 0 and 
                  rsi_aligned[i] < 50 and
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Any condition fails
            if (kama_slope[i] <= 0 or 
                rsi_aligned[i] <= 50 or
                chop_aligned[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Any condition fails
            if (kama_slope[i] >= 0 or 
                rsi_aligned[i] >= 50 or
                chop_aligned[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals