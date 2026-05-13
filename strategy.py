#!/usr/bin/env python3
# 1d_KAMA_RSI_ChopFilter_TrendFollow_v2
# Hypothesis: On 1d timeframe, KAMA trend direction (adaptive trend filter) combined with RSI extreme
# reversals and Choppiness Index regime filter (CHOP > 61.8 = ranging) provides high-probability
# mean-reversion entries in ranging markets and trend continuation in trending markets.
# Weekly trend filter (price vs weekly EMA50) avoids counter-trend trades in strong trends.
# Designed for low-frequency, high-quality setups with clear risk control.

name = "1d_KAMA_RSI_ChopFilter_TrendFollow_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # KAMA trend efficiency ratio (10-period)
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Pad beginning with zeros
    er = np.concatenate([np.full(10, np.nan), er])

    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # Weekly trend filter (EMA50 on weekly close)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning
    rsi = np.concatenate([np.full(14, np.nan), rsi])

    # Choppiness Index (14-period)
    atr1 = np.abs(np.subtract(high[1:], low[:-1]))
    atr2 = np.abs(np.subtract(close[1:], close[:-1]))
    tr = np.concatenate([[np.nan], np.maximum(atr1, atr2)])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high - lowest_low) != 0,
                    100 * np.log10(np.sum(atr) / (highest_high - lowest_low)) / np.log10(14),
                    50)
    chop = np.concatenate([np.full(13, np.nan), chop])

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend (price > KAMA) + RSI oversold (<30) + Chop > 61.8 (ranging)
            if close[i] > kama[i] and rsi[i] < 30 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (price < KAMA) + RSI overbought (>70) + Chop > 61.8 (ranging)
            elif close[i] < kama[i] and rsi[i] > 70 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or trend turns bearish (price < KAMA)
            if rsi[i] > 70 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or trend turns bullish (price > KAMA)
            if rsi[i] < 30 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals