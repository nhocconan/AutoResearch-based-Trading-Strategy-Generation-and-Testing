#!/usr/bin/env python3
# 12h_Donchian_Breakout_Volume_Trend
# Hypothesis: Donchian(20) breakout on 12h timeframe with volume confirmation and 1d EMA trend filter.
# Works in bull via upward breakouts above resistance and in bear via downward breakdowns below support.
# Volume filter ensures breakouts have conviction. EMA filter avoids counter-trend trades.
# Target: 20-40 trades/year (80-160 total over 4 years) to stay within fee limits.

name = "12h_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
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

    # Donchian channels: 20-period high and low
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values

    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume filter: >1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(lookback, n):
        # Skip if any required value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + volume spike + above 1d EMA50
            if (close[i] > donchian_high[i] and 
                volume[i] > vol_avg_20[i] * 1.8 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + volume spike + below 1d EMA50
            elif (close[i] < donchian_low[i] and 
                  volume[i] > vol_avg_20[i] * 1.8 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel (below midpoint) or trend change
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel (above midpoint) or trend change
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals