#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use daily Camarilla pivot R1/S1 levels as breakout triggers.
# Enter long when price breaks above R1 with 1d EMA50 uptrend and volume spike.
# Enter short when price breaks below S1 with 1d EMA50 downtrend and volume spike.
# Exit when price returns to the central pivot point (PP) to avoid reversals.
# Camarilla levels are derived from prior day's range and work well in trending markets.
# Combined with 1d trend filter and volume confirmation to reduce false signals.
# Target: 20-30 trades/year on 4h to minimize fee drag while capturing strong moves.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for each day
    # Based on previous day's OHLC
    range_1d = high_1d - low_1d
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s1 = pp - (high_1d - low_1d) * 1.1 / 4.0

    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.8x 24-period average (to filter weak moves)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 + price > 1d EMA50 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_24[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 + price < 1d EMA50 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_24[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below pivot point (mean reversion to fair value)
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back above pivot point
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals