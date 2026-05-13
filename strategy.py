#!/usr/bin/env python3
# 6h_WeeklyPivot_RangeBreakout_Volume
# Hypothesis: 6-hour breakouts from weekly pivot range (PP to R1/S1) with volume confirmation.
# Uses weekly pivot levels as dynamic support/resistance. In ranging markets, price often
# respects weekly pivot bands; breakouts with volume indicate institutional participation.
# Works in bull/bear by only trading breakouts in direction of weekly trend (above/below PP).
# Weekly trend defined by price vs weekly open. Avoids whipsaws in low-volume conditions.

name = "6h_WeeklyPivot_RangeBreakout_Volume"
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

    # Get weekly data for pivot points and trend
    df_weekly = get_htf_data(prices, '1w')

    # Calculate weekly pivot points: (H+L+C)/3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high

    # Align weekly pivot levels to 6h (updated only after weekly bar closes)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)

    # Weekly trend: price above/below weekly open (held until next weekly close)
    weekly_trend = np.where(weekly_close >= weekly_open, 1, -1)  # 1=up, -1=down
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend)

    # Volume spike: current volume > 1.5x 20-period average (to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 with volume spike and weekly uptrend
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and 
                weekly_trend_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume spike and weekly downtrend
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and 
                  weekly_trend_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below PP or weekly trend turns down
            if close[i] < pp_aligned[i] or weekly_trend_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above PP or weekly trend turns up
            if close[i] > pp_aligned[i] or weekly_trend_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals