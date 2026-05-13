#!/usr/bin/env python3
# 6h_WeeklyPivot_1dTrend_VolumeBreakout
# Hypothesis: Trade 6-hour breakouts from weekly pivot levels (R3/S3, R4/S4) with daily trend filter and volume confirmation.
# Long when price breaks above weekly R3 during daily uptrend with volume spike.
# Short when price breaks below weekly S3 during daily downtrend with volume spike.
# Exit when price crosses weekly pivot point (PP) or trend reverses.
# Weekly pivots provide structural support/resistance; daily trend avoids counter-trend whipsaws.
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).
# Target: 15-30 trades/year per symbol to minimize fee drag.

name = "6h_WeeklyPivot_1dTrend_VolumeBreakout"
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

    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot points: PP = (H+L+C)/3, R3 = H + 2*(PP-L), S3 = L - 2*(H-PP)
    # Using prior week's values (already closed due to get_htf_data)
    pp_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    r3_1w = df_1w['high'] + 2 * (pp_1w - df_1w['low'])
    s3_1w = df_1w['low'] - 2 * (df_1w['high'] - pp_1w)
    r4_1w = df_1w['high'] + 3 * (pp_1w - df_1w['low'])
    s4_1w = df_1w['low'] - 3 * (df_1w['high'] - pp_1w)

    # Align weekly pivots to 6h timeframe (wait for weekly close)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w.values)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w.values)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w.values)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w.values)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w.values)

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    # Daily EMA50 for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: >1.5x 24-period average (4 days of 6h bars)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R3 with daily uptrend and volume spike
            # Use R4 as stronger breakout confirmation, but enter at R3 break
            if close[i] > r3_1w_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > vol_avg_24[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with daily downtrend and volume spike
            elif close[i] < s3_1w_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > vol_avg_24[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly PP or trend turns down
            if close[i] < pp_1w_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly PP or trend turns up
            if close[i] > pp_1w_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals