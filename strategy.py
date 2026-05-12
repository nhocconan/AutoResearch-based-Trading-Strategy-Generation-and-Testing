#!/usr/bin/env python3
"""
12h_DonchianBreakout_20_1dTrend_Volume
Hypothesis: Trade Donchian(20) breakout on 12h timeframe with 1d trend filter and volume confirmation.
Donchian channels provide clear breakout signals, 1d trend ensures directionality, volume confirms institutional participation.
Works in bull/bear by following daily trend direction, avoids whipsaws via trend filter. Targets 20-50 trades/year.
"""

name = "12h_DonchianBreakout_20_1dTrend_Volume"
timeframe = "12h"
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

    # Get daily data for trend and volume filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate daily EMA20 for trend filter
    daily_close = df_1d['close'].values
    daily_ema20 = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_ema20_aligned = align_htf_to_ltf(prices, df_1d, daily_ema20)

    # Calculate daily volume average for volume filter
    daily_volume = df_1d['volume'].values
    daily_vol_avg = pd.Series(daily_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_vol_avg_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_avg)

    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(daily_ema20_aligned[i]) or np.isnan(daily_vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above upper Donchian + daily trend up + volume confirmation
            if (close[i] > high_20[i] and 
                close[i] > daily_ema20_aligned[i] and 
                volume[i] > 1.5 * daily_vol_avg_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian + daily trend down + volume confirmation
            elif (close[i] < low_20[i] and 
                  close[i] < daily_ema20_aligned[i] and 
                  volume[i] > 1.5 * daily_vol_avg_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below lower Donchian
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above upper Donchian
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals