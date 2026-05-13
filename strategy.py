#!/usr/bin/env python3
# 6h_WeeklyPivot_DailyTrend_Breakout
# Hypothesis: Breakout above weekly pivot resistance (R4) in bullish daily trend or below weekly pivot support (S4) in bearish daily trend, with volume confirmation. Uses weekly pivot levels from weekly close/open and daily trend filter to capture momentum in both bull and bear markets while maintaining low trade frequency.

name = "6h_WeeklyPivot_DailyTrend_Breakout"
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

    # Weekly pivot points (using weekly OHLC)
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    # Weekly range
    weekly_range = df_1w['high'] - df_1w['low']
    # Weekly support/resistance levels
    weekly_r4 = weekly_pivot + 1.5 * weekly_range  # R4
    weekly_s4 = weekly_pivot - 1.5 * weekly_range  # S4
    weekly_r3 = weekly_pivot + 1.0 * weekly_range  # R3
    weekly_s3 = weekly_pivot - 1.0 * weekly_range  # S3
    # Align to 6h timeframe with 1-bar delay (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4.values)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4.values)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3.values)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3.values)

    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume filter: >2.0x 24-period average (4 days worth of 6h bars)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above weekly R4 + daily EMA34 uptrend + volume spike
            if (close[i] > weekly_r4_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below weekly S4 + daily EMA34 downtrend + volume spike
            elif (close[i] < weekly_s4_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly pivot or volatility drop
            if close[i] < weekly_pivot_aligned[i] or volume[i] < vol_avg_24[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly pivot or volatility drop
            if close[i] > weekly_pivot_aligned[i] or volume[i] < vol_avg_24[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals