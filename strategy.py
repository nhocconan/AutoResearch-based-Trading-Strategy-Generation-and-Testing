#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakouts from Camarilla R1/S1 levels with 1d EMA trend filter and volume spike capture momentum moves while minimizing false breakouts.
# Works in bull markets via breakouts and in bear via mean reversion touches of the middle line (EMA).
# Target: 20-40 trades/year per symbol.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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

    # Get 1d data for trend filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate Camarilla levels (R1, S1, P) using previous day's range
    # We need to calculate for each 4h bar based on previous day's OHLC
    # Since we don't have daily data aligned to 4h, we'll use a rolling window of 96 bars (4h * 24h / 4h = 6, but we'll use previous day)
    # Instead, we'll calculate based on the highest high and lowest low of the previous 24h period (6 * 4h bars)
    lookback = 6  # 6 * 4h = 24h
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1)  # previous 24h
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1)   # previous 24h
    prev_close = pd.Series(close).shift(1)  # previous close

    # Calculate pivot and Camarilla levels
    pivot = (highest_high + lowest_low + prev_close) / 3
    range_hl = highest_high - lowest_low
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)

    # Volume confirmation: volume > 1.5x 24-period average (more conservative)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(lookback, n):
        if np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 + 1d uptrend + volume spike
            if close[i] > r1[i] and close[i] > ema34_1d_aligned[i] and volume[i] > vol_avg_24[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 + 1d downtrend + volume spike
            elif close[i] < s1[i] and close[i] < ema34_1d_aligned[i] and volume[i] > vol_avg_24[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below pivot or 1d trend turns down
            if close[i] < pivot[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above pivot or 1d trend turns up
            if close[i] > pivot[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals