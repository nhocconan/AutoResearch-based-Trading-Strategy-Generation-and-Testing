#!/usr/bin/env python3
# 6h_Weekly_Pivot_Volume_Squeeze
# Hypothesis: Price breaking weekly pivot levels (R1/S1) with volume confirmation and 1d trend filter captures institutional flow.
# Weekly pivot provides structural support/resistance from prior week, filtering false breakouts.
# Works in bull markets via breakouts above R1, in bear via breakdowns below S1.
# Target: 15-25 trades/year per symbol.

name = "6h_Weekly_Pivot_Volume_Squeeze"
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

    # Get weekly data for pivot calculation (call once before loop)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's OHLC
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # We use the completed weekly bar's data
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    # Calculate pivot for each completed weekly bar
    pivot_w = (weekly_high + weekly_low + weekly_close) / 3.0
    r1_w = 2 * pivot_w - weekly_low
    s1_w = 2 * pivot_w - weekly_high
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # 1d EMA20 for trend
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or \
           np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above weekly R1 + 1d uptrend + volume spike
            if close[i] > r1_w_aligned[i] and close[i] > ema20_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below weekly S1 + 1d downtrend + volume spike
            elif close[i] < s1_w_aligned[i] and close[i] < ema20_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below weekly pivot or 1d trend turns down
            if close[i] < pivot_w_aligned[i] or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above weekly pivot or 1d trend turns up
            if close[i] > pivot_w_aligned[i] or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals