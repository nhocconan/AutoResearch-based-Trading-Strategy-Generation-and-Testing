#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Use Camarilla R3/S3 levels from daily as breakout levels with 1w trend filter.
# Enter long when price breaks above R3 with 1w EMA uptrend and volume spike.
# Enter short when price breaks below S3 with 1w EMA downtrend and volume spike.
# Exit when price returns to the central pivot (CP) level.
# Camarilla levels provide institutional support/resistance, weekly trend filters reduce whipsaws,
# and volume confirms breakout strength. Designed for 12h to target 12-37 trades/year.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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

    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels for each daily bar
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # CP = (high + low + close) / 3
    range_1d = high_1d - low_1d
    r3 = close_1d + 1.1 * range_1d
    s3 = close_1d - 1.1 * range_1d
    cp = (high_1d + low_1d + close_1d) / 3.0

    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    cp_aligned = align_htf_to_ltf(prices, df_1d, cp)

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: volume > 1.8x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(cp_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R3 + price > weekly EMA34 + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and
                volume[i] > vol_avg_30[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S3 + price < weekly EMA34 + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and
                  volume[i] > vol_avg_30[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below central pivot (CP)
            if close[i] < cp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back above central pivot (CP)
            if close[i] > cp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals