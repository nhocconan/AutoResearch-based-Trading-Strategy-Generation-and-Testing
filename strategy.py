#!/usr/bin/env python3
"""
6h_Keltner_Channel_Trend_Reversal
Hypothesis: Price tends to revert from extreme deviations in volatile markets.
Uses Keltner Channel (2.0 ATR) on 6h with 1d trend filter: long when price touches lower band in uptrend, short when touches upper band in downtrend.
Exit on middle band (EMA20) touch or opposite band touch. Designed for mean reversion in ranging markets and pullbacks in trends.
Targets 15-30 trades/year to minimize fee impact.
"""

name = "6h_Keltner_Channel_Trend_Reversal"
timeframe = "6h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate ATR(10) for Keltner Channel
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Keltner Channel: EMA20 ± 2*ATR
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper = ema20 + 2.0 * atr
    lower = ema20 - 2.0 * atr

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema20[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches lower band in 1d uptrend
            if low[i] <= lower[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches upper band in 1d downtrend
            elif high[i] >= upper[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches upper band or middle band (EMA20)
            if high[i] >= upper[i] or abs(close[i] - ema20[i]) < 0.001 * close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches lower band or middle band (EMA20)
            if low[i] <= lower[i] or abs(close[i] - ema20[i]) < 0.001 * close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals