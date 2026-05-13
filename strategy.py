#!/usr/bin/env python3
# 1d_WR_Extreme_S1S3_Reversal_1wTrend
# Hypothesis: On 1d timeframe, trade reversals at Williams %R extremes (WR < -80 for long, WR > -20 for short) 
# when price touches weekly S1 or R3 levels, with weekly trend filter (price above/below weekly EMA50).
# Uses 1d Williams %R for mean reversion signals and weekly S1/R3 as dynamic support/resistance.
# Weekly trend filter ensures trading with higher timeframe momentum. Designed for low frequency
# to minimize fee drag in ranging and trending markets (works in bull/bear via mean reversion + trend filter).
# Target: 10-25 trades/year per symbol.

name = "1d_WR_Extreme_S1S3_Reversal_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values

    # Get weekly data for S1, R3 levels and trend filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate weekly pivot points (standard floor method)
    # Pivot = (H + L + C) / 3
    # S1 = (2 * Pivot) - High
    # R3 = Pivot + 2 * (High - Low)
    # Note: Using previous week's OHLC to avoid look-ahead
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_s1 = (2 * weekly_pivot) - prev_week_high
    weekly_r3 = weekly_pivot + 2 * (prev_week_high - prev_week_low)

    # Align weekly S1/R3 to daily timeframe
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)

    # Weekly trend filter: EMA50 on weekly close
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # 14-period lookback
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, ((highest_high - close) / hh_ll) * -100, -50)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # Start after WR lookback period
        # Skip if any required value is NaN
        if (np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Williams %R oversold (< -80) AND price at or below weekly S1 support
            if (williams_r[i] < -80 and 
                low[i] <= weekly_s1_aligned[i] and 
                close[i] > ema50_1w_aligned[i]):  # Only long in weekly uptrend
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20) AND price at or above weekly R3 resistance
            elif (williams_r[i] > -20 and 
                  high[i] >= weekly_r3_aligned[i] and 
                  close[i] < ema50_1w_aligned[i]):  # Only short in weekly downtrend
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R exits oversold OR price crosses above weekly pivot (take profit)
            if williams_r[i] > -50 or close[i] > weekly_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R exits overbought OR price crosses below weekly pivot
            if williams_r[i] < -50 or close[i] < weekly_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals