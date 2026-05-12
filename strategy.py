#!/usr/bin/env python3
# 6h_1W_1D_MultiTF_Trend_Volume
# Hypothesis: Use weekly trend direction and daily volume confirmation for 6h breakouts at key levels.
# Weekly trend provides long-term bias (works in bull/bear), daily volume confirms institutional participation.
# Breakouts at 6h Donchian channels with weekly trend alignment and volume spike.
# Targets 12-37 trades/year on 6h timeframe.

name = "6h_1W_1D_MultiTF_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for trend determination
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Weekly trend: price above/below 21-period EMA
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)

    # Daily volume confirmation: current volume > 2x average of last 20 days
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)

    # 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter
        weekly_uptrend = close[i] > ema_21_1w_aligned[i]
        weekly_downtrend = close[i] < ema_21_1w_aligned[i]

        # Daily volume confirmation
        volume_spike = volume[i] > (2.0 * vol_ma_20_aligned[i])

        if position == 0:
            # LONG: Break above 6h Donchian high with weekly uptrend and volume spike
            if (close[i] > high_20[i] and weekly_uptrend and volume_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below 6h Donchian low with weekly downtrend and volume spike
            elif (close[i] < low_20[i] and weekly_downtrend and volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Donchian low or weekly trend turns down
            if close[i] < low_20[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Donchian high or weekly trend turns up
            if close[i] > high_20[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals