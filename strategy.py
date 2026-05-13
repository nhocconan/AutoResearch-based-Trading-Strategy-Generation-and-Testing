#!/usr/bin/env python3
# 1d_Keltner_Breakout_1wTrend
# Hypothesis: Keltner Channel breakout with weekly trend filter captures strong trends
# while avoiding whipsaws. Enter long when price closes above upper KC with weekly
# uptrend (price > weekly EMA50). Enter short when price closes below lower KC with
# weekly downtrend. Exit on opposite KC touch. Designed for low-frequency, high-conviction
# trades on daily timeframe to minimize fee drag and work in both bull and bear markets.
# Target: 10-25 trades/year per symbol.

name = "1d_Keltner_Breakout_1wTrend"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Keltner Channel (20, 2.0) on daily data
    atr_period = 20
    kc_multiplier = 2.0
    ema_period = 20

    # True Range
    tr0 = np.abs(high - low)
    tr1 = np.abs(high - np.roll(close, 1))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    tr[0] = tr0[0]

    # ATR
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period:i])

    # EMA (middle line)
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values

    # Upper and Lower KC
    kc_upper = ema + kc_multiplier * atr
    kc_lower = ema - kc_multiplier * atr

    # Weekly EMA50 for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(max(atr_period, ema_period) + 1, n):
        # Skip if data not ready
        if np.isnan(atr[i]) or np.isnan(ema[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above upper KC with weekly uptrend
            if close[i] > kc_upper[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below lower KC with weekly downtrend
            elif close[i] < kc_lower[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Touch or cross below lower KC
            if close[i] < kc_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Touch or cross above upper KC
            if close[i] > kc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals