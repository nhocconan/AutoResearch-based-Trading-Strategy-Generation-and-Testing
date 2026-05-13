#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_ChopFilter
# Hypothesis: Use 1d KAMA for trend direction, 1d RSI for overbought/oversold conditions, and weekly Chop Index for regime filter.
# Enter long when KAMA is rising, RSI < 30 (oversold), and weekly Chop Index > 61.8 (ranging market) for mean reversion.
# Enter short when KAMA is falling, RSI > 70 (overbought), and weekly Chop Index > 61.8 (ranging market).
# Exit when RSI returns to neutral (40-60 range) or Chop Index < 38.2 (trending market).
# Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
# Target: 10-25 trades/year per symbol.

name = "1d_KAMA_Direction_RSI_ChopFilter"
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

    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = |Close - Close[10]| / Sum(|Close - Close[1]|, 10)
    # SSC = [ER * (Fastest SC - Slowest SC) + Slowest SC]^2
    # KAMA = Previous KAMA + SSC * (Close - Previous KAMA)
    change = np.abs(np.diff(close_1d, k=10))
    volatility = np.sum(np.abs(np.diff(close_1d, k=1)), axis=1)  # Simplified for 10-period
    er = np.where(volatility != 0, change / volatility, 0)
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    ssc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Initialize
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + ssc[i] * (close_1d[i] - kama[i-1])

    # Calculate RSI (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Get 1w data for Chop Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate Chop Index (14-period)
    # True Range = max(high-low, |high-close_prev|, |low-close_prev|)
    # Chop = 100 * log10(sum(TR,14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high_1w - low_1w
    tr2 = np.abs(np.subtract(high_1w, np.roll(close_1w, 1)))
    tr3 = np.abs(np.subtract(low_1w, np.roll(close_1w, 1)))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = np.full_like(close_1w, np.nan)
    for i in range(13, len(true_range)):
        atr_sum[i] = np.sum(true_range[i-13:i+1])
    max_high = np.full_like(close_1w, np.nan)
    min_low = np.full_like(close_1w, np.nan)
    for i in range(13, len(close_1w)):
        max_high[i] = np.max(high_1w[i-13:i+1])
        min_low[i] = np.min(low_1w[i-13:i+1])
    chop = np.full_like(close_1w, np.nan)
    for i in range(13, len(close_1w)):
        if max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # Avoid division by zero

    # Align indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Wait for sufficient data
        # Skip if data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]

        if position == 0:
            # LONG: KAMA rising, RSI oversold (<30), Chop > 61.8 (ranging)
            if kama_rising and rsi_aligned[i] < 30 and chop_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI overbought (>70), Chop > 61.8 (ranging)
            elif kama_falling and rsi_aligned[i] > 70 and chop_aligned[i] > 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (40-60) or Chop < 38.2 (trending)
            if (rsi_aligned[i] >= 40 and rsi_aligned[i] <= 60) or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (40-60) or Chop < 38.2 (trending)
            if (rsi_aligned[i] >= 40 and rsi_aligned[i] <= 60) or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals