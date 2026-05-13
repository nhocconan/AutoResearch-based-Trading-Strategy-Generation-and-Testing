#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Use 1d Camarilla pivot levels (R3/S3) as breakout levels with confirmation from 1d EMA trend and volume spikes.
# Enter long when price breaks above R3 with volume spike and 1d EMA34 uptrend.
# Enter short when price breaks below S3 with volume spike and 1d EMA34 downtrend.
# Exit when price returns to the previous day's close (C level).
# Uses 6h timeframe with 1d trend filter to balance trade frequency and win rate.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 12-37 trades/year per symbol.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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

    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for previous day
    # P = (H + L + C) / 3
    # Range = H - L
    # S3 = C - (Range * 1.1 / 4)
    # R3 = C + (Range * 1.1 / 4)
    P = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d

    S3 = close_1d - (rng * 1.1 / 4)
    R3 = close_1d + (rng * 1.1 / 4)

    # Align pivot levels to 6h timeframe (use previous day's levels)
    s3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, R3)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Align previous day's close (C level) for exit
    c_prev = np.roll(close_1d, 1)
    c_prev[0] = np.nan
    c_aligned = align_htf_to_ltf(prices, df_1d, c_prev)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(c_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R3 with volume spike and 1d EMA uptrend
            if close[i] > r3_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3 with volume spike and 1d EMA downtrend
            elif close[i] < s3_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to previous day's close (C level)
            if close[i] <= c_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to previous day's close (C level)
            if close[i] >= c_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals