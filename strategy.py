#!/usr/bin/env python3
# 1d_KAMA_RSI_ChopFilter_TrendFollow
# Hypothesis: On daily timeframe, KAMA identifies adaptive trend direction, RSI filters extreme
# values in trending markets, and Choppiness Index (14) confirms regime (trending when < 38.2).
# This combination avoids whipsaws in ranging markets while capturing strong trends in both
# bull and bear phases. Low-frequency signals reduce fee drag, and daily timeframe avoids
# noise of lower timeframes.

name = "1d_KAMA_RSI_ChopFilter_TrendFollow"
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

    # KAMA (Kaufman Adaptive Moving Average) - adaptive trend
    # ER = Efficiency Ratio = |close - close[10]| / sum(|close - close[1]|) over 10
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = np.nan  # first 10 values invalid
    volatility = np.abs(np.diff(close, prepend=np.nan))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = change / volatility_sum
    er[0:10] = np.nan
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI(14)
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Choppiness Index (14)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Above KAMA, RSI not overbought, trending market (CHOP < 38.2)
            if close[i] > kama[i] and rsi[i] < 70 and chop[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # SHORT: Below KAMA, RSI not oversold, trending market (CHOP < 38.2)
            elif close[i] < kama[i] and rsi[i] > 30 and chop[i] < 38.2:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Below KAMA or RSI overbought or choppy market
            if close[i] < kama[i] or rsi[i] > 75 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Above KAMA or RSI oversold or choppy market
            if close[i] > kama[i] or rsi[i] < 25 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals