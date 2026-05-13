#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) from 1d act as support/resistance. Breakouts above R1 or below S1 with 1d trend alignment (price > EMA50 for long, < EMA50 for short) and volume confirmation (>1.5x 30-period average) capture institutional moves. Works in bull via upward breakouts and bear via downward breakdowns. Target: 15-25 trades/year.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Calculate Camarilla pivot levels (R1, S1) from previous 12h bar's OHLC
    def camarilla_levels(h, l, c):
        # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
        r1 = c + (h - l) * 1.1 / 12
        s1 = c - (h - l) * 1.1 / 12
        return r1, s1

    # Shift by 1 to use previous bar's levels (no look-ahead)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    for i in range(1, n):
        r1[i], s1[i] = camarilla_levels(high[i-1], low[i-1], close[i-1])

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume filter: >1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 + price > EMA50 (1d uptrend) + volume spike
            if (close[i] > r1[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 + price < EMA50 (1d downtrend) + volume spike
            elif (close[i] < s1[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or trend reverses (price < EMA50)
            if (close[i] < r1[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or trend reverses (price > EMA50)
            if (close[i] > s1[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals