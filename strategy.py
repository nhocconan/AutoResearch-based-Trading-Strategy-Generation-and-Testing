#!/usr/bin/env python3
"""
6h_WeeklyPivot_DailyTrend_Filter
Hypothesis: Use weekly pivot points as structural support/resistance and daily trend filter to enter long/short on 6f breakouts. In bull markets, buy near weekly S1/S2 with daily uptrend; in bear markets, sell near weekly R1/R2 with daily downtrend. Weekly pivots provide multi-week context while daily trend filters avoid counter-trend trades, reducing false breakouts and whipsaws. Targets 20-50 trades/year for low fee drag.
"""

name = "6h_WeeklyPivot_DailyTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values

    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)

    # Align weekly pivots to 6f timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)

    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    daily_close = df_daily['close'].values

    # Daily EMA50 for trend filter
    ema50_daily = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(ema50_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above S1 and daily uptrend (strong bullish setup)
            # Or price above S2 with momentum (stronger signal)
            if ((close[i] > s1_aligned[i] and close[i] > ema50_daily_aligned[i]) or
                (close[i] > s2_aligned[i] and close[i] > ema50_daily_aligned[i] * 1.01)):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below R1 and daily downtrend (strong bearish setup)
            # Or price below R2 with momentum (stronger signal)
            elif ((close[i] < r1_aligned[i] and close[i] < ema50_daily_aligned[i]) or
                  (close[i] < r2_aligned[i] and close[i] < ema50_daily_aligned[i] * 0.99)):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or daily trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or daily trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals