#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_Filter_v2
# Hypothesis: KAMA identifies trend direction, RSI filters entry timing, and Chop determines regime.
# In trending markets (Chop < 38.2): enter long when KAMA up + RSI > 50, short when KAMA down + RSI < 50.
# In ranging markets (Chop > 61.8): long when RSI < 30, short when RSI > 70.
# Exit on opposite RSI extreme. Daily timeframe to minimize trades (<20/year) and avoid fee drag.
# Uses weekly trend filter to ensure alignment with higher timeframe momentum.

name = "1d_KAMA_RSI_Chop_Filter_v2"
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

    # KAMA (adaptive moving average) - trend identification
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros(n)
    for i in range(10, n):
        if np.sum(volatility[i-9:i+1]) > 0:
            er[i] = np.sum(change[i-9:i+1]) / np.sum(volatility[i-9:i+1])
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI (14-period) - momentum oscillator
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[i-13:i+1])
            avg_loss[i] = np.mean(loss[i-13:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Chop Index (14-period) - regime detection
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros(n)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    chop = np.zeros(n)
    for i in range(14, n):
        if highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(np.sum(tr[i-13:i+1]) / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50

    # Weekly EMA (34-period) - higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Trending market (Chop < 38.2): follow KAMA trend with RSI filter
            if chop[i] < 38.2:
                if kama[i] > kama[i-1] and rsi[i] > 50 and close[i] > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif kama[i] < kama[i-1] and rsi[i] < 50 and close[i] < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (Chop > 61.8): mean reversion at RSI extremes
            elif chop[i] > 61.8:
                if rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: RSI overbought in range, or trend reversal in trend
            if chop[i] > 61.8 and rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            elif chop[i] < 38.2 and kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold in range, or trend reversal in trend
            if chop[i] > 61.8 and rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            elif chop[i] < 38.2 and kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals