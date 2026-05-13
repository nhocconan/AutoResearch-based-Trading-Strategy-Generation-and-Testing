#!/usr/bin/env python3
# 1d_Donchian_Breakout_VolumeTrend
# Hypothesis: Daily Donchian channel breakouts with volume confirmation and weekly trend filter.
# Works in bull/bear: long when price breaks above upper band with volume and weekly uptrend,
# short when breaks below lower band with volume and weekly downtrend.
# Designed for low frequency (10-25 trades/year) to minimize fee drag.

name = "1d_Donchian_Breakout_VolumeTrend"
timeframe = "1d"
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

    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate daily Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian band with volume and weekly uptrend
            if (close[i] > high_20[i] and 
                volume_spike[i] and 
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian band with volume and weekly downtrend
            elif (close[i] < low_20[i] and 
                  volume_spike[i] and 
                  close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian band
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian band
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals