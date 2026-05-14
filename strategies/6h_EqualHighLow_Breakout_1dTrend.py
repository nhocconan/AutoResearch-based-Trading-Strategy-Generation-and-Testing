#!/usr/bin/env python3
# 6h_EqualHighLow_Breakout_1dTrend
# Hypothesis: Identify equal highs/lows (liquidity pools) on 1d and trade breakouts.
# Equal highs form resistance, equal lows form support. Breakouts with volume
# and 1d EMA trend filter capture institutional moves. Works in bull/bear by
# following the dominant 1d trend. Low turnover expected (~20-40/year).

name = "6h_EqualHighLow_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for equal highs/lows and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate equal highs and lows on 1d (within 0.5% tolerance)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    equal_high = np.zeros(len(df_1d), dtype=bool)
    equal_low = np.zeros(len(df_1d), dtype=bool)

    for i in range(2, len(df_1d)):
        # Equal high: current high within 0.5% of previous high
        if abs(high_1d[i] - high_1d[i-1]) / high_1d[i-1] < 0.005:
            equal_high[i] = True
        # Equal low: current low within 0.5% of previous low
        if abs(low_1d[i] - low_1d[i-1]) / low_1d[i-1] < 0.005:
            equal_low[i] = True

    # Get the most recent equal high/low levels
    eq_high_level = np.full(len(df_1d), np.nan)
    eq_low_level = np.full(len(df_1d), np.nan)
    last_eq_high = np.nan
    last_eq_low = np.nan
    for i in range(len(df_1d)):
        if equal_high[i]:
            last_eq_high = high_1d[i]
        if equal_low[i]:
            last_eq_low = low_1d[i]
        eq_high_level[i] = last_eq_high
        eq_low_level[i] = last_eq_low

    # Align equal high/low levels to 6h timeframe
    eq_high_aligned = align_htf_to_ltf(prices, df_1d, eq_high_level)
    eq_low_aligned = align_htf_to_ltf(prices, df_1d, eq_low_level)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(eq_high_aligned[i]) or np.isnan(eq_low_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above equal high with volume spike and 1d EMA50 uptrend
            if close[i] > eq_high_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below equal low with volume spike and 1d EMA50 downtrend
            elif close[i] < eq_low_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below equal high (failed breakout)
            if close[i] < eq_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above equal low (failed breakout)
            if close[i] > eq_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals