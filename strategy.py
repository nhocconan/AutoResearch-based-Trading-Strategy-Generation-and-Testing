#!/usr/bin/env python3
# 1d_Weekly_Pivot_Reversal_1wTrend_Filter
# Hypothesis: Buy when price touches weekly S1 support in uptrend, sell when touches weekly R1 resistance in downtrend.
# Uses weekly pivot levels for mean reversion within the trend, filtering counter-trend trades.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Low frequency: only 1-2 signals per week, avoiding overtrading.

name = "1d_Weekly_Pivot_Reversal_1wTrend_Filter"
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

    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    # Use previous week's data to avoid look-ahead (current week still forming)
    p_high = np.roll(df_1w['high'].values, 1)
    p_low = np.roll(df_1w['low'].values, 1)
    p_close = np.roll(df_1w['close'].values, 1)

    # Calculate weekly pivot points
    pivot = (p_high + p_low + p_close) / 3.0
    range_val = p_high - p_low
    R1 = pivot + (range_val * 1.1 / 4)
    S1 = pivot - (range_val * 1.1 / 4)

    # Align weekly pivot levels to daily timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)

    # Get weekly EMA20 for trend filter (longer-term trend)
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup for weekly indicators
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches or goes below S1 support in weekly uptrend
            if low[i] <= S1_aligned[i] and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or goes above R1 resistance in weekly downtrend
            elif high[i] >= R1_aligned[i] and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches weekly pivot or trend turns down
            if high[i] >= pivot[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly pivot or trend turns up
            if low[i] <= pivot[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals