#!/usr/bin/env python3
# 12h_Donchian_Breakout_Volume_1dTrend
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter.
# Long when price breaks above 20-period high with volume > 1.5x average and price > 1d EMA50.
# Short when price breaks below 20-period low with volume > 1.5x average and price < 1d EMA50.
# Exit on opposite Donchian breakout or trend reversal. Designed for 15-30 trades/year per symbol.

name = "12h_Donchian_Breakout_Volume_1dTrend"
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
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Donchian channels (20-period) on 12h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 1.5x average of last 4 periods (2 days)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1d EMA50
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]

        if position == 0:
            # LONG: Price breaks above 20-period high + volume + uptrend
            if close[i] > high_20[i] and volume_ok[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low + volume + downtrend
            elif close[i] < low_20[i] and volume_ok[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low OR trend down
            if close[i] < low_20[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high OR trend up
            if close[i] > high_20[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals