#!/usr/bin/env python3
# 12h_Donchian_Breakout_20_D1Trend_Volume
# Hypothesis: 12h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
# Donchian channels provide clear breakout levels; EMA200 ensures trend alignment.
# Volume adds conviction. Designed for 12-37 trades/year, works in bull/bear via trend filter.

name = "12h_Donchian_Breakout_20_D1Trend_Volume"
timeframe = "12h"
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
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d EMA200 trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    # 12h Donchian(20) - using current 12h bar's high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 1.5x average of last 4 periods
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1d EMA200
        price_above_ema = close[i] > ema_200_1d_aligned[i]
        price_below_ema = close[i] < ema_200_1d_aligned[i]

        if position == 0:
            # LONG: Close breaks above Donchian high AND uptrend AND volume
            if close[i] > donchian_high[i] and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Donchian low AND downtrend AND volume
            elif close[i] < donchian_low[i] and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls back below Donchian low OR trend turns down
            if close[i] < donchian_low[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises back above Donchian high OR trend turns up
            if close[i] > donchian_high[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals