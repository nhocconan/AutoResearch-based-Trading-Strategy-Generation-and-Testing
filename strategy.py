#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: KAMA trend direction combined with RSI momentum and Choppiness regime filter captures sustained moves while avoiding whipsaws in ranging markets. Works in both bull and bear by adapting to trend strength via KAMA and filtering choppy conditions.
"""

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
    volume = prices['volume'].values

    # KAMA for trend direction
    close_series = pd.Series(close)
    change = close_series.diff().abs()
    volatility = close_series.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.finfo(float).eps)
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_series = pd.Series(kama)

    # RSI for momentum
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))

    # Choppiness Index for regime filter
    atr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising + RSI > 50 + Chop < 61.8 (trending)
            if kama[i] > kama[i-1] and rsi[i] > 50 and chop[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling + RSI < 50 + Chop < 61.8 (trending)
            elif kama[i] < kama[i-1] and rsi[i] < 50 and chop[i] < 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turning down OR Chop > 61.8 (choppy)
            if kama[i] < kama[i-1] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turning up OR Chop > 61.8 (choppy)
            if kama[i] > kama[i-1] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals