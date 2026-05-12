#!/usr/bin/env python3
# 1D_KAMA_20_RSI_14_CHOPPINESS_14_FILTER
# Hypothesis: KAMA(20) trend direction + RSI(14) extreme + Choppiness Index(14) regime filter on 1d timeframe.
# KAMA adapts to market noise, reducing false signals in ranging markets.
# RSI identifies overbought/oversold conditions for mean reversion entries.
# Choppiness Index filters for trending markets (CHOP < 38.2) to avoid whipsaws.
# Works in bull/bear markets by following KAMA direction only when RSI is extreme and market is trending.
# Targets BTC/ETH with tight entry to avoid whipsaw and reduce trade frequency.

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

    # Get 1w data for Choppiness Index calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate True Range (TR)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Calculate max/high and min/low over 14 periods
    max_hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values

    # Calculate Choppiness Index: CHOP = 100 * log10(sum(TR,14) / (maxHH - minLL)) / log10(14)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_hl = max_hh - min_ll
    chop = 100 * np.log10(sum_tr / range_hl) / np.log10(14)
    chop[range_hl == 0] = 50  # Avoid division by zero

    # Align Choppiness Index to lower timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)

    # Get 1d data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[-1]|, 10)
    change = np.abs(np.concatenate([[np.nan], close_1d[1:] - close_1d[:-1]]))
    er_num = np.abs(np.concatenate([[np.nan]*10, close_1d[10:] - close_1d[:-10]]))
    er_den = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    er = np.where(er_den != 0, er_num / er_den, 0)
    # Smoothing Constants: SC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # KAMA: kama[i] = kama[i-1] + SC * (close[i] - kama[i-1])
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # Align KAMA to lower timeframe (same timeframe, so identity)
    kama_aligned = kama  # Already 1d

    # Calculate RSI(14)
    delta = np.concatenate([[np.nan], close_1d[1:] - close_1d[:-1]])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[0] = 50  # First value neutral

    # Align RSI to lower timeframe (same timeframe, so identity)
    rsi_aligned = rsi  # Already 1d

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA up (uptrend), RSI oversold (<30), trending market (CHOP < 38.2)
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 30 and 
                chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down (downtrend), RSI overbought (>70), trending market (CHOP < 38.2)
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA down (trend change) OR RSI overbought (>70)
            if (close[i] < kama_aligned[i] or rsi_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA up (trend change) OR RSI oversold (<30)
            if (close[i] > kama_aligned[i] or rsi_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals