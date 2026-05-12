#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Daily Camarilla pivot levels (R1/S1) breakout with weekly EMA200 trend filter and volume confirmation (>1.5x 20-day average). Long when price breaks above R1 with close > weekly EMA200 and volume spike; short when price breaks below S1 with close < weekly EMA200 and volume spike. Exit when price returns to daily pivot (PP). Targets 15-25 trades/year to minimize fee decay and work in both bull/bear via trend filter.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate daily Camarilla pivot levels
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (high + low + close) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1 = close + (high - low) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = close - (high - low) * 1.1 / 12.0

    # Weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(pp[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 + close > weekly EMA200 + volume spike
            if (high[i] > r1[i] and 
                close[i] > ema200_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + close < weekly EMA200 + volume spike
            elif (low[i] < s1[i] and 
                  close[i] < ema200_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to or below pivot point
            if close[i] <= pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to or above pivot point
            if close[i] >= pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals