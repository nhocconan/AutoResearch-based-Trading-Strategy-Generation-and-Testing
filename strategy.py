#!/usr/bin/env python3
# 160107: 6h_Donchian_Breakout_20_1wTrend_WeeklyPivot_Confirmation
# Hypothesis: Combines Donchian breakout with weekly trend filter and weekly pivot levels to capture strong trends while avoiding false breakouts. Works in bull/bear by following higher timeframe trend. Weekly pivot provides additional confluence for entry/exit. Targets 15-35 trades/year on 6h.

name = "6h_Donchian_Breakout_20_1wTrend_WeeklyPivot_Confirmation"
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

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Weekly trend: SMA50
    sma_50_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)

    # Weekly pivot levels (from previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_point = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pivot_point - low_1w
    s1 = 2 * pivot_point - high_1w
    r2 = pivot_point + (high_1w - low_1w)
    s2 = pivot_point - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot_point - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot_point)

    # Align weekly levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)

    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Donchian high + weekly uptrend + above weekly R1 + volume
            if (close[i] > donchian_high[i] and 
                close[i] > sma_50_1w_aligned[i] and
                close[i] > r1_aligned[i] and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low + weekly downtrend + below weekly S1 + volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < sma_50_1w_aligned[i] and
                  close[i] < s1_aligned[i] and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below Donchian low OR below weekly S1
            if close[i] < donchian_low[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above Donchian high OR above weekly R1
            if close[i] > donchian_high[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals