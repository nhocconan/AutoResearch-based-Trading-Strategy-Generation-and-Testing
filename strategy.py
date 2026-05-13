#!/usr/bin/env python3
# 1h_4h1d_Trend_1h_Entry_Volume
# Hypothesis: Use 4h EMA20 and 1d EMA50 for trend alignment (both must agree) and 1h volume spike for entry.
# Long when price > 4h EMA20 > 1d EMA50 and volume > 2x 20-period average; short when opposite.
# Exit when either trend breaks. Designed for low trade frequency (<30/year) to avoid fee drag.
# Works in bull (trend follows) and bear (avoids counter-trend trades via dual timeframe filter).

name = "1h_4h1d_Trend_1h_Entry_Volume"
timeframe = "1h"
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

    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    ema20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume filter: >2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price above 4h EMA20 > 1d EMA50 (strong uptrend) + volume spike
            if (close[i] > ema20_4h_aligned[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.20
                position = 1
            # SHORT: price below 4h EMA20 < 1d EMA50 (strong downtrend) + volume spike
            elif (close[i] < ema20_4h_aligned[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below 4h EMA20 or 4h EMA20 drops below 1d EMA50
            if (close[i] < ema20_4h_aligned[i] or ema20_4h_aligned[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price breaks above 4h EMA20 or 4h EMA20 rises above 1d EMA50
            if (close[i] > ema20_4h_aligned[i] or ema20_4h_aligned[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals