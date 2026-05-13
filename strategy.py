#!/usr/bin/env python3
# 6h_WeeklyPivot_MeanReversion_1dTrend
# Hypothesis: Fade extreme daily closes at weekly pivot levels (R4/S4) with 1d trend filter.
# In bull markets, buy dips to weekly S3/S4 in uptrend; in bear markets, sell rallies to weekly R3/R4 in downtrend.
# Uses weekly pivot points (classic formula) calculated from prior week's OHLC.
# Mean reversion expected at extreme weekly levels with trend alignment reducing false signals.
# Target: 15-30 trades/year per symbol to minimize fee drag on 6h timeframe.

name = "6h_WeeklyPivot_MeanReversion_1dTrend"
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

    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')

    # Calculate weekly pivot points: P = (H+L+C)/3, R1=2P-L, S1=2P-H, etc.
    # Using prior week's values to avoid look-ahead
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values

    # Pivot point and support/resistance levels
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r4 = r3 + (weekly_high - weekly_low)
    s4 = s3 - (weekly_high - weekly_low)

    # Align weekly pivot levels to 6h timeframe (using prior week's values)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')

    # Trend filter: 1d EMA50 (only trade in direction of higher timeframe trend)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at or below weekly S4 in uptrend (mean reversion long)
            if (close[i] <= s4_aligned[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at or above weekly R4 in downtrend (mean reversion short)
            elif (close[i] >= r4_aligned[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point or trend turns down
            if (close[i] >= pp_aligned[i] or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point or trend turns up
            if (close[i] <= pp_aligned[i] or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals