#!/usr/bin/env python3
# 12h_Donchian_20_Breakout_1dTrend_Volume
# Hypothesis: Use 12h Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
# This strategy captures breakouts in the direction of the daily trend, reducing false signals.
# Volume confirmation ensures momentum behind the breakout. Designed for 12-37 trades/year per symbol.
# Works in both bull and bear markets via the trend filter.

name = "12h_Donchian_20_Breakout_1dTrend_Volume"
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

    # Get 1d data for trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Donchian channels from previous 1d bar (20-period high/low)
    # Use rolling window on 1d data, then shift by 1 to avoid look-ahead
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values

    # Align Donchian levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)

    # Volume confirmation: current volume > 1.5x average of last 4 periods (2 hours)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above upper Donchian AND uptrend AND volume
            if close[i] > high_20_aligned[i] and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian AND downtrend AND volume
            elif close[i] < low_20_aligned[i] and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below lower Donchian OR trend turns down
            if close[i] < low_20_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above upper Donchian OR trend turns up
            if close[i] > high_20_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals