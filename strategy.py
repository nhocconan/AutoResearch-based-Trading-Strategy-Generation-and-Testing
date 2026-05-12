#!/usr/bin/env python3
# 6h_1d_12h_VolumeBreakout_Trend
# Hypothesis: 6h volume breakout (price closes above/below previous day's high/low) with 12h trend filter and 1d volume confirmation.
# Works in bull markets (breakouts continue upward) and bear markets (breakdowns continue downward) by following the 12h trend.
# Volume confirmation filters out low-probability breakouts. Designed for 12-30 trades/year on 6h.

name = "6h_1d_12h_VolumeBreakout_Trend"
timeframe = "6h"
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

    # Get 1d data for breakout levels and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # Previous day's high and low for breakout levels
    prev_1d_high = np.roll(df_1d['high'].values, 1)
    prev_1d_low = np.roll(df_1d['low'].values, 1)
    prev_1d_high[0] = df_1d['high'].values[0]
    prev_1d_low[0] = df_1d['low'].values[0]

    # Align breakout levels to 6h timeframe
    breakout_high = align_htf_to_ltf(prices, df_1d, prev_1d_high)
    breakout_low = align_htf_to_ltf(prices, df_1d, prev_1d_low)

    # 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    # 1d volume confirmation: current volume > 1.5x average of last 20 days
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(breakout_high[i]) or np.isnan(breakout_low[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter
        bullish_trend = close[i] > ema_12h_aligned[i]
        bearish_trend = close[i] < ema_12h_aligned[i]

        # Volume confirmation
        volume_ok = volume[i] > (1.5 * vol_ma_1d_aligned[i])

        if position == 0:
            # LONG: Price closes above previous day's high with bullish 12h trend and volume confirmation
            if close[i] > breakout_high[i] and close[i-1] <= breakout_high[i-1] and bullish_trend and volume_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below previous day's low with bearish 12h trend and volume confirmation
            elif close[i] < breakout_low[i] and close[i-1] >= breakout_low[i-1] and bearish_trend and volume_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below previous day's low or 12h trend turns bearish
            if close[i] < breakout_low[i] and close[i-1] >= breakout_low[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above previous day's high or 12h trend turns bullish
            if close[i] > breakout_high[i] and close[i-1] <= breakout_high[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals