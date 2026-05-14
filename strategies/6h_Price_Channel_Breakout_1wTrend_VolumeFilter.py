#!/usr/bin/env python3
# 6h_Price_Channel_Breakout_1wTrend_VolumeFilter
# Hypothesis: Price channel breakouts (Donchian 20) combined with weekly trend filter and volume confirmation
# capture trending moves with low whipsaw. Works in both bull (breakout above upper channel in uptrend) and
# bear (breakdown below lower channel in downtrend) markets. Weekly trend ensures alignment with higher-timeframe
# momentum, reducing false signals. Volume filter confirms breakout strength. Target: 15-30 trades/year.

name = "6h_Price_Channel_Breakout_1wTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate weekly EMA20 for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate Donchian channels (20-period) on 6h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian channel with volume spike and weekly uptrend
            if close[i] > highest_high[i] and volume_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian channel with volume spike and weekly downtrend
            elif close[i] < lowest_low[i] and volume_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters the channel (below midpoint) or weekly trend turns down
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] < midpoint or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters the channel (above midpoint) or weekly trend turns up
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] > midpoint or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals