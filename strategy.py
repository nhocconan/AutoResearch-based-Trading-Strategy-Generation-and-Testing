#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Use Camarilla R3/S3 levels from daily pivot for breakout entries, confirmed by 1d EMA50 trend and volume spikes.
# This combines price channel breakout with trend alignment and volume confirmation for high-probability setups.
# Designed to work in both bull and bear markets by filtering counter-trend trades and avoiding overtrading.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Calculate Camarilla levels from previous day's range
    df_1d = get_htf_data(prices, '1d')
    phigh = np.roll(df_1d['high'].values, 1)
    plow = np.roll(df_1d['low'].values, 1)
    pclose = np.roll(df_1d['close'].values, 1)
    # First value will be invalid due to roll, but we'll handle via min_periods later

    range_val = phigh - plow
    R3 = pclose + (range_val * 1.1 / 4)
    S3 = pclose - (range_val * 1.1 / 4)

    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume spike detection: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            if close[i] > R3_aligned[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            elif close[i] < S3_aligned[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            if close[i] < S3_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            if close[i] > R3_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals