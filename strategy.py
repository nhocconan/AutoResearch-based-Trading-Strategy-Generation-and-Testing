#!/usr/bin/env python3
# 1d_WeeklyPivot_PriceAction_Reversal
# Hypothesis: Price reversals at weekly pivot levels with volume confirmation capture mean-reversion moves in both bull and bear markets.
# Weekly pivots act as institutional support/resistance; price reacting off these levels with volume shows rejection.
# Entry: Long when price bounces off weekly S1/S2/S3 with volume spike; Short when price rejects at weekly R1/R2/R3 with volume spike.
# Exit: Mean reversion to weekly pivot point (PP) or opposite pivot level to avoid overstaying.
# Target: 10-25 trades/year on 1d to stay within optimal range while capturing significant reversals.

name = "1d_WeeklyPivot_PriceAction_Reversal"
timeframe = "1d"
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

    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)

    # Calculate weekly pivot points: PP = (H+L+C)/3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly support and resistance levels
    weekly_r1 = 2 * weekly_pp - weekly_low
    weekly_r2 = weekly_pp + (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pp - weekly_low)
    weekly_s1 = 2 * weekly_pp - weekly_high
    weekly_s2 = weekly_pp - (weekly_high - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pp)

    # Align weekly pivot levels to daily timeframe (wait for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price bounces off weekly support with volume spike
            if ((close[i] >= s1_aligned[i] * 0.995 and close[i] <= s1_aligned[i] * 1.005) or
                (close[i] >= s2_aligned[i] * 0.995 and close[i] <= s2_aligned[i] * 1.005) or
                (close[i] >= s3_aligned[i] * 0.995 and close[i] <= s3_aligned[i] * 1.005)) and \
               volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price rejects at weekly resistance with volume spike
            elif ((close[i] >= r1_aligned[i] * 0.995 and close[i] <= r1_aligned[i] * 1.005) or
                  (close[i] >= r2_aligned[i] * 0.995 and close[i] <= r2_aligned[i] * 1.005) or
                  (close[i] >= r3_aligned[i] * 0.995 and close[i] <= r3_aligned[i] * 1.005)) and \
                 volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to weekly pivot or resistance
            if close[i] >= pp_aligned[i] or close[i] >= r1_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to weekly pivot or support
            if close[i] <= pp_aligned[i] or close[i] <= s1_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals