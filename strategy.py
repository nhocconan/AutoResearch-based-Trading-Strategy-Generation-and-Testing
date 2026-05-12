# 6h_PivotBreakout_WeeklyTrend_VolumeFilter
# Hypothesis: On 6h timeframe, use weekly pivot points from previous week as key support/resistance.
# Enter long when price breaks above weekly R1 with volume > 2x average and weekly trend up (price > weekly EMA50).
# Enter short when price breaks below weekly S1 with volume > 2x average and weekly trend down (price < weekly EMA50).
# Weekly trend filter reduces false breakouts in ranging markets. Targets 20-50 trades per year to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via short breakdowns.

name = "6h_PivotBreakout_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot points and trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values

    # Calculate weekly pivot points (standard floor trader's method)
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    pp = (high_w + low_w + close_w) / 3
    weekly_r1 = (2 * pp) - low_w
    weekly_s1 = (2 * pp) - high_w

    # Use previous week's levels (shift by 1)
    weekly_r1_prev = np.roll(weekly_r1, 1)
    weekly_s1_prev = np.roll(weekly_s1, 1)
    weekly_r1_prev[0] = np.nan
    weekly_s1_prev[0] = np.nan

    # Align weekly pivot levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_w, weekly_r1_prev)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_w, weekly_s1_prev)

    # Weekly EMA50 for trend filter
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)

    # Volume confirmation: volume > 2x 24-period average (approx 12 hours on 6h chart)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema50_w_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly R1 + weekly uptrend + volume spike
            if (close[i] > weekly_r1_aligned[i] and 
                close[i] > ema50_w_aligned[i] and 
                volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S1 + weekly downtrend + volume spike
            elif (close[i] < weekly_s1_aligned[i] and 
                  close[i] < ema50_w_aligned[i] and 
                  volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly S1 OR trend turns down
            if close[i] < weekly_s1_aligned[i] or close[i] < ema50_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly R1 OR trend turns up
            if close[i] > weekly_r1_aligned[i] or close[i] > ema50_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals