#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_ChopFilter
# Hypothesis: Use Kaufman's Adaptive Moving Average (KAMA) on 1d for trend direction,
# RSI(14) on 4h for momentum confirmation, and Choppiness Index on 4h to avoid ranging markets.
# Enter long when KAMA trending up, RSI > 50, and Chop < 38.2 (trending).
# Enter short when KAMA trending down, RSI < 50, and Chop < 38.2.
# Exit when KAMA changes direction or Chop > 61.8 (ranging).
# Designed to work in both bull (trend following) and bear (trend following short).
# Target: 20-30 trades/year per symbol.

name = "4h_KAMA_Direction_RSI_ChopFilter"
timeframe = "4h"
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

    # Get 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate KAMA on 1d
    # ER = |Close - Close(10)| / Sum(|Close - Close(1)|, 9)
    # SC = [ER * (2/(2+1) - 2/(30+1)) + 2/(30+1)]^2
    # KAMA = KAMA(1) + SC * (Close - KAMA(1))
    change = np.abs(np.diff(close_1d, 10))  # |Close - Close(10)|
    volatility = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    # For first 9 periods, volatility is not defined, so we'll handle with slicing
    er = np.full_like(close_1d, np.nan)
    for i in range(10, len(close_1d)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smooth constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = np.full_like(close_1d, np.nan)
    for i in range(10, len(close_1d)):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start with close at period 9
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_1d = kama

    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)

    # RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # For first 14 periods, RSI is not defined, but we'll use 50 as neutral
    rsi[:14] = 50

    # Choppiness Index on 4h (14-period)
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    # ATR = average TR over 14 periods
    # Chop = 100 * log10(sum(TR, 14) / (ATR * 14)) / log10(14)
    tr = np.zeros(n)
    for i in range(n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1]) if i > 0 else 0
        lc = abs(low[i] - close[i-1]) if i > 0 else 0
        tr[i] = max(hl, hc, lc)
    atr = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    sum_tr = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            sum_tr[i] = np.sum(tr[1:15])
        else:
            sum_tr[i] = sum_tr[i-1] + tr[i] - tr[i-14]
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if atr[i] > 0:
            chop[i] = 100 * np.log10(sum_tr[i] / (atr[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # neutral if no volatility

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA up, RSI > 50, Chop < 38.2 (trending)
            if close[i] > kama_aligned[i] and rsi[i] > 50 and chop[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down, RSI < 50, Chop < 38.2 (trending)
            elif close[i] < kama_aligned[i] and rsi[i] < 50 and chop[i] < 38.2:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA down OR Chop > 61.8 (ranging)
            if close[i] < kama_aligned[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA up OR Chop > 61.8 (ranging)
            if close[i] > kama_aligned[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals