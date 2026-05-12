#!/usr/bin/env python3
"""
6h_Liquidity_Clearing_1dTrend
Hypothesis: On 6h timeframe, identify daily liquidity zones (previous day's high/low) and look for price to clear these levels with volume confirmation in the direction of the 1d trend. In trending markets, liquidity sweeps often precede continuation. Uses 1d EMA50 for trend filter and 1d liquidity zones for entry/exit. Targets 15-30 trades per year to minimize fee drag while capturing trend continuation moves.
"""

name = "6h_Liquidity_Clearing_1dTrend"
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

    # Get 1d data for trend filter and liquidity zones
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Previous day's high/low for liquidity zones (use prior 1d bar's high/low)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    high_1d_prev[0] = np.nan  # first value invalid
    low_1d_prev[0] = np.nan

    # Align 1d liquidity zones to 6h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d_prev)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d_prev)

    # Volume confirmation: volume > 1.5x 20-period average (approx 6 hours)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price clears above 1d high + 1d uptrend + volume spike
            if (close[i] > high_1d_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price clears below 1d low + 1d downtrend + volume spike
            elif (close[i] < low_1d_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price fails to hold above 1d low OR trend turns down
            if close[i] < low_1d_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price fails to hold below 1d high OR trend turns up
            if close[i] > high_1d_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals