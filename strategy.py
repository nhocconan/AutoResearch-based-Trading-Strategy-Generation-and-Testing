#!/usr/bin/env python3
# 1d_KAMA_Trend_with_RSI_and_Choppy_Filter
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to determine trend direction on daily timeframe.
# Enter long when price crosses above KAMA and RSI < 70 (not overbought) and market is not choppy (Choppiness Index > 38.2).
# Enter short when price crosses below KAMA and RSI > 30 (not oversold) and market is not choppy.
# Exit when price crosses back across KAMA or RSI reaches extreme levels.
# Designed for low-frequency signals (7-25 trades/year) to minimize fee drag, works in both bull and bear markets via trend following with momentum and regime filters.

name = "1d_KAMA_Trend_with_RSI_and_Choppy_Filter"
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

    # Get weekly data for chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # KAMA on daily close
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(slow+1) - 2/(fast+1)) + 2/(fast+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[length-1] = close[length-1]
        for i in range(length, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    kama_vals = kama(close, 10, 2, 30)

    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with close

    # Choppiness Index (14) on weekly
    def choppiness_index(high, low, close, length=14):
        atr = np.zeros_like(high)
        tr1 = high - low
        tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=length, min_periods=length).mean().values
        highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        ci = 100 * np.log10(atr * length / (highest_high - lowest_low)) / np.log10(length)
        return ci

    chop_vals = choppiness_index(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_vals)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if (np.isnan(kama_vals[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend and momentum conditions
        price_above_kama = close[i] > kama_vals[i]
        price_below_kama = close[i] < kama_vals[i]
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        not_choppy = chop_aligned[i] > 38.2  # trending market

        if position == 0:
            # LONG: price above KAMA, RSI not overbought, trending market
            if price_above_kama and rsi_not_overbought and not_choppy:
                signals[i] = 0.25
                position = 1
            # SHORT: price below KAMA, RSI not oversold, trending market
            elif price_below_kama and rsi_not_oversold and not_choppy:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price below KAMA or RSI overbought or choppy
            if not price_above_kama or not rsi_not_overbought or not not_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price above KAMA or RSI oversold or choppy
            if not price_below_kama or not rsi_not_oversold or not not_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals