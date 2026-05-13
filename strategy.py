#!/usr/bin/env python3
# 6h_Monthly_Pivot_Breakout_1dTrend_Volume
# Hypothesis: Monthly pivot levels act as significant support/resistance on 6h timeframe.
# Breakout above monthly R1 with 1d uptrend and volume surge captures institutional momentum.
# Breakdown below monthly S1 with 1d downtrend and volume surge captures breakdowns.
# Monthly pivots are less noisy than daily and capture longer-term structure, working in both bull/bear markets.

name = "6h_Monthly_Pivot_Breakout_1dTrend_Volume"
timeframe = "6h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Get monthly data for pivot calculation
    df_1M = get_htf_data(prices, '1M')
    high_1M = df_1M['high'].values
    low_1M = df_1M['low'].values
    close_1M = df_1M['close'].values

    # Calculate Monthly Pivot levels (similar to Camarilla but using standard pivot)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_1M = (high_1M + low_1M + close_1M) / 3.0
    r1_1M = 2 * pivot_1M - low_1M
    s1_1M = 2 * pivot_1M - high_1M

    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1M, r1_1M)
    s1_aligned = align_htf_to_ltf(prices, df_1M, s1_1M)

    # Volume spike: volume > 2.0 * 20-period average (~6.6 days at 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + breakout above monthly R1 + volume spike
            if close[i] > ema34_1d_aligned[i] and close[i] > r1_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + breakdown below monthly S1 + volume spike
            elif close[i] < ema34_1d_aligned[i] and close[i] < s1_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below monthly S1 or trend turns bearish
            if close[i] < s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above monthly R1 or trend turns bullish
            if close[i] > r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals