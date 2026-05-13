#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_1dVolume
# Hypothesis: Camarilla R1/S1 breakout on 12h with 1w trend filter (price above/below 50-period EMA) and 1d volume spike.
# Works in bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend).
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_1dVolume"
timeframe = "12h"
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

    # Calculate Camarilla levels from previous day
    def camarilla_levels(h, l, c):
        # Previous day's range
        rng = h - l
        # Camarilla levels
        R4 = c + rng * 1.1 / 2
        R3 = c + rng * 1.1 / 4
        R2 = c + rng * 1.1 / 6
        R1 = c + rng * 1.1 / 12
        S1 = c - rng * 1.1 / 12
        S2 = c - rng * 1.1 / 6
        S3 = c - rng * 1.1 / 4
        S4 = c - rng * 1.1 / 2
        return R1, S1

    # Need previous day's OHLC for Camarilla
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan

    R1, S1 = camarilla_levels(prev_high, prev_low, prev_close)

    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    vol_avg_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 + price above 1w EMA50 + volume spike
            if (close[i] > R1[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_20_aligned[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 + price below 1w EMA50 + volume spike
            elif (close[i] < S1[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_20_aligned[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or trend changes
            if (close[i] < R1[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or trend changes
            if (close[i] > S1[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals