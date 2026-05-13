#!/usr/bin/env python3
"""
12h_Pivot_Breakout_Trend
Hypothesis: On 12H timeframe, combine weekly trend filter with daily Pivot Point (Classic) support/resistance levels.
Enter long when price breaks above weekly uptrend + R1 pivot level with volume confirmation.
Enter short when price breaks below weekly downtrend + S1 pivot level with volume confirmation.
Exit on trend reversal or price crossing opposite pivot level.
Uses weekly trend to avoid counter-trend trades, pivot levels for structure, and volume to filter breakouts.
Designed for low trade frequency (~20-40/year) to minimize fee drag while capturing meaningful moves in both bull and bear markets.
"""

name = "12h_Pivot_Breakout_Trend"
timeframe = "12h"
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

    # Get daily data for Pivot Points (classic formula)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily Pivot Points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2.0 * pivot - low_1d
    s1 = 2.0 * pivot - high_1d

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    # Weekly trend: 20-period EMA on weekly close
    weekly_ema20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = close_1w > weekly_ema20
    weekly_downtrend = close_1w < weekly_ema20

    # Volume confirmation: current volume > 1.8 * 30-period average (higher threshold for lower frequency)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)

    # Align HTF arrays to 12L timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)  # align volume MA for comparison

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup for weekly EMA and volume MA
        # Get aligned values
        p = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        up_trend = weekly_uptrend_aligned[i]
        down_trend = weekly_downtrend_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        vol_spike = volume_spike[i]  # volume_spike is already LTF aligned

        # Skip if any critical value is NaN
        if (np.isnan(p) or np.isnan(r1_val) or np.isnan(s1_val) or
            np.isnan(vol_ma_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly uptrend + price breaks above R1 + volume spike
            if (up_trend and close[i] > r1_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + price breaks below S1 + volume spike
            elif (down_trend and close[i] < s1_val and vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below pivot OR weekly trend turns down
            if (close[i] < p or not up_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above pivot OR weekly trend turns up
            if (close[i] > p or not down_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals