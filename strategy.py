#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_1dTrend_VolumeConfirm
Hypothesis: On 6h timeframe, buy when price breaks above weekly pivot R1 with volume >1.5x average and 1d EMA50 trending up; sell when price breaks below weekly pivot S1 with volume >1.5x average and 1d EMA50 trending down. Uses weekly pivot levels from prior week for structure and 1d trend filter to avoid false breakouts in ranging markets. Targets 10-20 trades per year to minimize fee drag and improve generalization across bull/bear cycles.
"""
name = "6h_WeeklyPivot_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtrader.libs.mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot levels (prior week)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values

    # Calculate weekly pivot points (standard formula)
    pivot = (high_w + low_w + close_w) / 3
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)

    # Use previous week's levels (shift by 1)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r2_prev = np.roll(r2, 1)
    s2_prev = np.roll(s2, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    r2_prev[0] = np.nan
    s2_prev[0] = np.nan

    # Align weekly levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_w, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1_prev)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2_prev)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2_prev)

    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    close_d = df_d['close'].values

    # 1d EMA50 for trend filter
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)

    # Volume confirmation: volume > 1.5x 50-period average (~200 hours / 6h = ~3.3 days)
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_d_aligned[i]) or np.isnan(vol_avg_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly R1 + 1d uptrend + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_d_aligned[i] and 
                volume[i] > vol_avg_50[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S1 + 1d downtrend + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_d_aligned[i] and 
                  volume[i] > vol_avg_50[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly S1 OR trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema50_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly R1 OR trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema50_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals