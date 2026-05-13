#!/usr/bin/env python3
# 12h_PriceChannel_Breakout_1dTrend_Volume
# Hypothesis: 12-hour price channel breakouts (Donchian 20) with 1-day trend filter and volume spike confirmation.
# The strategy captures breakouts from key intraday support/resistance levels with trend alignment.
# Donchian channel provides clear breakout levels, while 1-day EMA50 filters for trend direction.
# Volume confirmation ensures breakouts are supported by participation.
# Designed for 12-37 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.

name = "12h_PriceChannel_Breakout_1dTrend_Volume"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate Donchian channel (20-period) for 12h timeframe
    # Upper channel = highest high over last 20 periods
    # Lower channel = lowest low over last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Trend filter: 1-day EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above upper Donchian channel in uptrend with volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower Donchian channel in downtrend with volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian channel or trend turns down
            if close[i] < lowest_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian channel or trend turns up
            if close[i] > highest_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals