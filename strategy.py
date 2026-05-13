#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1wTrend
# Hypothesis: Camarilla pivot levels from 1d (R3/S3) act as key support/resistance.
# Breakouts above R3 or below S3 with volume confirmation and 1w EMA trend filter
# capture institutional moves. Works in bull/bear by following the dominant 1w trend.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

name = "6h_Camarilla_R3_S3_Breakout_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    r3 = np.full(len(df_1d), np.nan)
    s3 = np.full(len(df_1d), np.nan)
    r4 = np.full(len(df_1d), np.nan)
    s4 = np.full(len(df_1d), np.nan)

    for i in range(len(df_1d)):
        # Camarilla formula: range = high - low
        # R3 = close + (high - low) * 1.1 / 2
        # S3 = close - (high - low) * 1.1 / 2
        # R4 = close + (high - low) * 1.1
        # S4 = close - (high - low) * 1.1
        rng = high_1d[i] - low_1d[i]
        if rng <= 0:
            continue
        r3[i] = close_1d[i] + rng * 1.1 / 2
        s3[i] = close_1d[i] - rng * 1.1 / 2
        r4[i] = close_1d[i] + rng * 1.1
        s4[i] = close_1d[i] - rng * 1.1

    # Get most recent R3/S3 levels (carry forward until new level)
    r3_level = np.full(len(df_1d), np.nan)
    s3_level = np.full(len(df_1d), np.nan)
    last_r3 = np.nan
    last_s3 = np.nan
    for i in range(len(df_1d)):
        if not np.isnan(r3[i]):
            last_r3 = r3[i]
        if not np.isnan(s3[i]):
            last_s3 = s3[i]
        r3_level[i] = last_r3
        s3_level[i] = last_s3

    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_level)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_level)

    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)

    # Get 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R3 with volume spike and 1w EMA34 uptrend
            if close[i] > r3_aligned[i] and volume_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3 with volume spike and 1w EMA34 downtrend
            elif close[i] < s3_aligned[i] and volume_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below R3 (failed breakout) or reaches R4 (take profit)
            if close[i] < r3_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above S3 (failed breakout) or reaches S4 (take profit)
            if close[i] > s3_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals