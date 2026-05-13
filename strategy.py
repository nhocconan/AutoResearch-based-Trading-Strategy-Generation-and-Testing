#!/usr/bin/env python3
# 1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume
# Hypothesis: Use 1h for entry timing with Camarilla R1/S1 breakout, filtered by 4h trend (EMA50) and 1d volume spike.
# The 4h trend ensures directional bias, while 1d volume confirms institutional interest.
# Designed to work in both bull and bear markets by only taking trend-aligned breakouts and avoiding low-volume noise.
# Target: 15-30 trades/year per symbol with strict entry conditions to minimize fee drag.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume"
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

    # Session filter: 08:00 to 20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    # Get daily data for Camarilla levels (using prior day)
    df_1d = get_htf_data(prices, '1d')
    phigh = np.roll(df_1d['high'].values, 1)
    plow = np.roll(df_1d['low'].values, 1)
    pclose = np.roll(df_1d['close'].values, 1)
    range_val = phigh - plow
    R1 = pclose + (range_val * 1.1 / 6)
    S1 = pclose - (range_val * 1.1 / 6)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # Get 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Get 1d volume for spike detection (current vs 20-day average)
    vol_20d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_20d)
    volume_spike = df_1d['volume'].values > (2.0 * vol_20d_aligned)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 with volume spike and uptrend (4h EMA50)
            if close[i] > R1_aligned[i] and volume_spike_aligned[i] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 with volume spike and downtrend (4h EMA50)
            elif close[i] < S1_aligned[i] and volume_spike_aligned[i] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or trend turns down (close < 4h EMA50)
            if close[i] < S1_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or trend turns up (close > 4h EMA50)
            if close[i] > R1_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals