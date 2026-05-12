#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: Use Camarilla pivot levels (R1/S1) from 1d timeframe as dynamic support/resistance.
# Enter long when price breaks above R1 with volume confirmation (>2x 20-bar avg) and price above 1d EMA50 (uptrend).
# Enter short when price breaks below S1 with volume confirmation and price below 1d EMA50 (downtrend).
# Exit when price returns to the 1d EMA50 (mean reversion to trend) or on opposite signal.
# Uses 12h timeframe to limit trades (target: 50-150 over 4 years) and reduce fee drift.
# Works in bull/bear via EMA50 trend filter and volatility-adjusted breakout.

name = "12h_Camarilla_R1_S1_Breakout_Trend_Volume"
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

    # Get 1d data for Camarilla pivots and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels (R1, S1) from prior 1d bar
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    hl_range_1d = high_1d - low_1d
    r1_1d = close_1d + 1.1 * hl_range_1d / 12
    s1_1d = close_1d - 1.1 * hl_range_1d / 12

    # Align Camarilla levels to 12h (wait for 1d bar to close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # warmup for EMA50 and rolling
        # Skip if any required value is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price > R1 + volume spike + price > EMA50 (uptrend)
            if (close[i] > r1_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price < S1 + volume spike + price < EMA50 (downtrend)
            elif (close[i] < s1_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to EMA50 (mean reversion to trend)
            if close[i] <= ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to EMA50 (mean reversion to trend)
            if close[i] >= ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals