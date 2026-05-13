#!/usr/bin/env python3
# 4h_Donchian_20_Volume_Spike_1dTrend_EMA50_Trend
# Hypothesis: 4h Donchian(20) breakout with volume spike and 1d EMA50 trend filter.
# Works in bull markets by catching breakouts with momentum, and in bear markets by
# avoiding counter-trend trades via 1d EMA50 filter. Volume spike confirms breakout
# strength, reducing false signals. Designed for low trade frequency to avoid fee drag.

name = "4h_Donchian_20_Volume_Spike_1dTrend_EMA50_Trend"
timeframe = "4h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')

    # Donchian Channel (20) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_donchian = highest_high.values
    lower_donchian = lowest_low.values

    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (2.0 * vol_ma.values)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above upper Donchian with volume spike and uptrend (close > 1d EMA50)
            if (close[i] > upper_donchian[i] and 
                volume_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower Donchian with volume spike and downtrend (close < 1d EMA50)
            elif (close[i] < lower_donchian[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian or trend turns down
            if close[i] < lower_donchian[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian or trend turns up
            if close[i] > upper_donchian[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals