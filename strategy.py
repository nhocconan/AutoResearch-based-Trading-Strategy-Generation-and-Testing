#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1wTrend_1dVolume
# Hypothesis: Use 12h Donchian channel breakout with weekly trend filter and daily volume confirmation.
# Long when price breaks above 20-period 12h high with weekly close above weekly SMA40 and daily volume spike.
# Short when price breaks below 20-period 12h low with weekly close below weekly SMA40 and daily volume spike.
# Exit when price returns to 12h midline or volume drops below average.
# Weekly trend filter reduces whipsaw in bear markets, volume confirmation ensures breakout strength.

name = "12h_Donchian20_Breakout_1wTrend_1dVolume"
timeframe = "12h"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate 20-period Donchian channel on 12h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2

    # Calculate weekly SMA40 for trend filter
    sma_40_1w = pd.Series(df_1w['close']).rolling(window=40, min_periods=40).mean().values
    sma_40_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_40_1w)

    # Calculate daily volume average for volume filter
    vol_avg_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(sma_40_1w_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + weekly close above SMA40 (uptrend) + volume spike
            if (close[i] > high_max_20[i] and 
                close[i] > sma_40_1w_aligned[i] and
                volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + weekly close below SMA40 (downtrend) + volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < sma_40_1w_aligned[i] and
                  volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to midline or volume drops below average
            if (close[i] <= donchian_mid[i] or 
                volume[i] < vol_avg_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to midline or volume drops below average
            if (close[i] >= donchian_mid[i] or 
                volume[i] < vol_avg_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals