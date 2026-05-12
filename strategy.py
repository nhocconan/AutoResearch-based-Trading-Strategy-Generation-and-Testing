#!/usr/bin/env python3
# 1D_KAMA_20_RSI_14_CHOPPINESS_14_FILTER
# Hypothesis: KAMA(20) determines trend direction on daily timeframe, combined with RSI(14) for momentum and Choppiness Index(14) for regime filtering.
# KAMA adapts to market noise, reducing false signals in choppy markets. RSI confirms momentum strength.
# Choppiness Index filters out ranging markets (CHOP > 61.8) to avoid false breakouts.
# Works in both bull and bear markets by following KAMA trend direction with momentum confirmation.
# Targets BTC/ETH with low trade frequency to avoid fee drag.

name = "1D_KAMA_20_RSI_14_CHOPPINESS_14_FILTER"
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

    # Get weekly data for Choppiness Index calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate True Range and ATR for Choppiness Index
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index

    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Calculate Choppiness Index: 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low

    chop = np.full_like(atr, np.nan)
    valid = (atr_sum > 0) & (range_hl > 0)
    chop[valid] = 100 * np.log10(atr_sum[valid] / range_hl[valid]) / np.log10(14)

    # Align Choppiness Index to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)

    # Get daily data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio = |change| / sum(|changes|)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    abs_change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    er = np.full_like(close_1d, np.nan)
    change_sum = pd.Series(abs_change).rolling(window=10, min_periods=10).sum().values
    er = np.divide(change, change_sum, where=change_sum != 0)
    er[0] = 0  # First value

    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # Fast=2, Slow=30
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # Align KAMA to daily timeframe (already daily, but align for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)

    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # When no loss, RSI = 100
    rsi[avg_gain == 0] = 0    # When no gain, RSI = 0

    # Align RSI to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA, RSI > 50 (bullish momentum), and not choppy (CHOP <= 61.8)
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] <= 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, RSI < 50 (bearish momentum), and not choppy (CHOP <= 61.8)
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] <= 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR RSI < 40 (loss of momentum)
            if (close[i] < kama_aligned[i] or 
                rsi_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR RSI > 60 (loss of momentum)
            if (close[i] > kama_aligned[i] or 
                rsi_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals