#!/usr/bin/env python3
# 1d_KAMA_30_RSI_14_Chop_14
# Hypothesis: 1d strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
# RSI for momentum confirmation, and Choppiness Index to filter ranging markets.
# Trades only in trending regimes (Chop < 38.2) with KAMA direction and RSI momentum.
# Designed for low trade frequency (~10-25 trades/year) to minimize fee drag.
# Works in bull/bear markets by following adaptive trend and avoiding whipsaws in chop.

name = "1d_KAMA_30_RSI_14_Chop_14"
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

    # Get 1d data (already primary timeframe, but needed for indicators)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate KAMA (30) for trend direction
    # Efficiency Ratio (ER) = |net change| / sum(|changes|)
    change = np.abs(np.diff(close_1d))
    abs_change = np.abs(np.diff(close_1d))
    dir_10 = np.abs(close_1d[30:] - close_1d[:-30])
    vol_30 = np.nansum(np.abs(np.diff(close_1d.reshape(-1, 30))), axis=1)
    er = np.where(vol_30 > 0, dir_10 / vol_30, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[30] = close_1d[30]
    for i in range(31, len(close_1d)):
        kama[i] = kama[i-1] + sc[i-30] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)

    # Calculate RSI (14) for momentum
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)

    # Calculate Choppiness Index (14) for regime filter
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([0]), tr])  # align length
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) > 0, chop, 50)  # avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA, RSI > 50, and trending regime (Chop < 38.2)
            if (close[i] > kama_1d_aligned[i] and
                rsi_1d_aligned[i] > 50 and
                chop_1d_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, RSI < 50, and trending regime (Chop < 38.2)
            elif (close[i] < kama_1d_aligned[i] and
                  rsi_1d_aligned[i] < 50 and
                  chop_1d_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR chop becomes too high (ranging)
            if close[i] < kama_1d_aligned[i] or chop_1d_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR chop becomes too high (ranging)
            if close[i] > kama_1d_aligned[i] or chop_1d_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals