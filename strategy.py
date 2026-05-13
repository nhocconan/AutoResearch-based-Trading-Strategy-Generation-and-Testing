#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Breakout_1dTrend
# Hypothesis: Use 1d Camarilla pivot levels (S1/S2/S3/S4 and R1/R2/R3/R4) as
# support/resistance. Enter long when price breaks above R1 with volume and
# 1d EMA uptrend; enter short when price breaks below S1 with volume and
# 1d EMA downtrend. Exit when price returns to the mean (C level). Uses 12h
# timeframe for lower turnover. Works in bull by buying breakouts in uptrend
# and in bear by selling breakdowns in downtrend. Target 15-30 trades/year.

name = "12h_Camarilla_Pivot_Breakout_1dTrend"
timeframe = "12h"
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

    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for previous day
    # P = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 12)
    # S2 = C - (Range * 1.1 / 6)
    # S3 = C - (Range * 1.1 / 4)
    # S4 = C - (Range * 1.1 / 2)
    # R1 = C + (Range * 1.1 / 12)
    # R2 = C + (Range * 1.1 / 6)
    # R3 = C + (Range * 1.1 / 4)
    # R4 = C + (Range * 1.1 / 2)
    P = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d

    S1 = close_1d - (rng * 1.1 / 12)
    S2 = close_1d - (rng * 1.1 / 6)
    S3 = close_1d - (rng * 1.1 / 4)
    S4 = close_1d - (rng * 1.1 / 2)
    R1 = close_1d + (rng * 1.1 / 12)
    R2 = close_1d + (rng * 1.1 / 6)
    R3 = close_1d + (rng * 1.1 / 4)
    R4 = close_1d + (rng * 1.1 / 2)

    # Use S1 and R1 as primary breakout levels
    s1_level = S1
    r1_level = R1

    # Align pivot levels to 12h timeframe (use previous day's levels)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R1 with volume spike and 1d EMA uptrend
            if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and 1d EMA downtrend
            elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to pivot point (C level)
            # Use aligned P level from previous day
            P_prev = np.roll(P, 1)
            P_prev[0] = np.nan
            P_aligned = align_htf_to_ltf(prices, df_1d, P_prev)
            if not np.isnan(P_aligned[i]) and close[i] <= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to pivot point (C level)
            P_prev = np.roll(P, 1)
            P_prev[0] = np.nan
            P_aligned = align_htf_to_ltf(prices, df_1d, P_prev)
            if not np.isnan(P_aligned[i]) and close[i] >= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals