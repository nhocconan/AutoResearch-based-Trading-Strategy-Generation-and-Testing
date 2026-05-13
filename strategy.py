#!/usr/bin/env python3
# 1D_Donchian20_WeeklyTrend_Filter
# Hypothesis: On the daily timeframe, breakouts from the 20-period Donchian channel
# in the direction of the weekly trend (EMA50) with volume confirmation capture
# significant moves in both bull and bear markets. The weekly trend filter ensures
# we trade with the higher-timeframe momentum, reducing whipsaws. Volume confirmation
# adds conviction to breakouts. Target: 15-25 trades/year per symbol.

name = "1D_Donchian20_WeeklyTrend_Filter"
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
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Weekly trend: 50-period EMA on weekly close
    weekly_ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > weekly_ema50
    weekly_downtrend = close_1w < weekly_ema50

    # Daily Donchian channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned weekly values for current daily bar
        if i < len(weekly_uptrend):
            up_trend = weekly_uptrend[i]
            down_trend = weekly_downtrend[i]
        else:
            up_trend = False
            down_trend = False

        if i < len(volume_spike):
            vol_spike = volume_spike[i]
        else:
            vol_spike = False

        # Skip if any required data is not available
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly uptrend + price breaks above Donchian high + volume spike
            if (up_trend and 
                close[i] > highest_high_20[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + price breaks below Donchian low + volume spike
            elif (down_trend and 
                  close[i] < lowest_low_20[i] and vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian low or weekly trend changes to down
            if (close[i] < lowest_low_20[i] or not up_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian high or weekly trend changes to up
            if (close[i] > highest_high_20[i] or not down_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals