#!/usr/bin/env python3
# 6h_Weekly_Pivot_Breakout_Trend_Filter
# Hypothesis: Use weekly pivot levels as key support/resistance. Go long when price breaks above weekly R1 with
# weekly trend up (price > weekly EMA50), short when breaks below weekly S1 with weekly trend down.
# Weekly pivots provide structure from higher timeframe; breakouts indicate momentum shifts.
# Trend filter ensures we trade with the weekly momentum, reducing false breakouts in chop.
# Designed for 15-35 trades/year per symbol, works in both bull and bear via trend filter.

name = "6h_Weekly_Pivot_Breakout_Trend_Filter"
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

    # Get weekly data for pivot and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)

    # Weekly EMA50 trend filter
    ema_50_weekly = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)

    # Calculate weekly pivot points from previous weekly bar
    # Standard floor pivot: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    prev_weekly_high = df_weekly['high'].shift(1).values
    prev_weekly_low = df_weekly['low'].shift(1).values
    prev_weekly_close = df_weekly['close'].shift(1).values

    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    r1 = 2 * pivot - prev_weekly_low
    s1 = 2 * pivot - prev_weekly_high

    # Align pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)  # redundant but clear

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_weekly_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from weekly EMA50
        price_above_ema = close[i] > ema_50_weekly_aligned[i]
        price_below_ema = close[i] < ema_50_weekly_aligned[i]

        if position == 0:
            # LONG: Close breaks above weekly R1 AND weekly trend up
            if close[i] > r1_aligned[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below weekly S1 AND weekly trend down
            elif close[i] < s1_aligned[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls back below weekly R1 OR trend turns down
            if close[i] < r1_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises back above weekly S1 OR trend turns up
            if close[i] > s1_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals