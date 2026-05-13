#!/usr/bin/env python3
# 6h_WeeklyPivot_DailyTrend_Filter
# Hypothesis: Weekly pivot levels act as strong support/resistance on 6h timeframe.
# In bull markets, price tends to bounce from weekly S1/S2 and break above R1/R2.
# In bear markets, price tends to reject at weekly R1/R2 and break below S1/S2.
# We filter trades by daily trend (EMA50) to avoid counter-trend entries.
# Volume confirmation ensures breakouts/bounces have conviction.
# Target: 15-30 trades/year on 6h to stay within optimal range.

name = "6h_WeeklyPivot_DailyTrend_Filter"
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

    # Calculate 5-period ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=5, adjust=False, min_periods=5).mean().values

    # EMA50 for daily trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Get weekly data once
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)

    # Calculate weekly pivot points (standard formula)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)

    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(ema50[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price near weekly support AND daily uptrend AND volume spike
            near_support = (low[i] <= s1_aligned[i] * 1.002 or 
                          low[i] <= s2_aligned[i] * 1.002)
            daily_uptrend = close[i] > ema50[i]
            volume_spike = volume[i] > vol_avg_20[i] * 1.5

            if near_support and daily_uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price near weekly resistance AND daily downtrend AND volume spike
            elif (high[i] >= r1_aligned[i] * 0.998 or 
                  high[i] >= r2_aligned[i] * 0.998) and \
                 close[i] < ema50[i] and \
                 volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches weekly resistance or trend breaks
            if (high[i] >= r1_aligned[i] * 0.998 or 
                close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly support or trend breaks
            if (low[i] <= s1_aligned[i] * 1.002 or 
                close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals