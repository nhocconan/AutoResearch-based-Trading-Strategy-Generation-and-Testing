#!/usr/bin/env python3

# 1h_4h_1d_CombinedTrendBreakout
# Hypothesis: Combine 4h trend direction (SuperTrend) and 1d momentum (EMA crossover) with 1d volume confirmation.
# Enter long when 4h uptrend + 1d EMA(12) > EMA(26) + volume > 1.5x average; short when opposite.
# Exit when trend or momentum conditions fail. Uses 1h only for timing entries/exits.
# Designed for 15-35 trades/year to avoid fee drag, works in bull/bear via trend/momentum alignment.

name = "1h_4h_1d_CombinedTrendBreakout"
timeframe = "1h"
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

    # Get 4h data for trend (SuperTrend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate ATR for SuperTrend (10-period)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    hl2_4h = (high_4h + low_4h) / 2
    upper_band_4h = hl2_4h + (3.0 * atr_4h)
    lower_band_4h = hl2_4h - (3.0 * atr_4h)

    supertrend_4h = np.full(len(close_4h), np.nan)
    direction_4h = np.full(len(close_4h), 1)  # 1 for uptrend, -1 for downtrend
    supertrend_4h[0] = upper_band_4h[0]
    direction_4h[0] = 1

    for i in range(1, len(close_4h)):
        if close_4h[i-1] > supertrend_4h[i-1]:
            supertrend_4h[i] = max(lower_band_4h[i], supertrend_4h[i-1])
        else:
            supertrend_4h[i] = min(upper_band_4h[i], supertrend_4h[i-1])

        if close_4h[i] > supertrend_4h[i]:
            direction_4h[i] = 1
        else:
            direction_4h[i] = -1

    # Align 4h SuperTrend direction to 1h
    supertrend_4h_dir_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)

    # Get 1d data for momentum and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate 1d EMA(12) and EMA(26) for momentum
    ema12_1d = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema12_1d_aligned = align_htf_to_ltf(prices, df_1d, ema12_1d)
    ema26_1d_aligned = align_htf_to_ltf(prices, df_1d, ema26_1d)

    # Calculate 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)

    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Warmup for indicators
        # Skip if any required data is NaN
        if (np.isnan(supertrend_4h_dir_aligned[i]) or np.isnan(ema12_1d_aligned[i]) or
            np.isnan(ema26_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # 1d momentum: EMA(12) > EMA(26) = bullish, < = bearish
        bullish_momentum = ema12_1d_aligned[i] > ema26_1d_aligned[i]
        bearish_momentum = ema12_1d_aligned[i] < ema26_1d_aligned[i]

        # 1d volume confirmation: current volume > 1.5x average
        volume_ok = volume[i] > (1.5 * vol_ma_1d_aligned[i])

        if position == 0:
            # LONG: 4h uptrend + 1d bullish momentum + volume
            if (supertrend_4h_dir_aligned[i] == 1 and bullish_momentum and volume_ok):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend + 1d bearish momentum + volume
            elif (supertrend_4h_dir_aligned[i] == -1 and bearish_momentum and volume_ok):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h downtrend OR 1d bearish momentum OR no volume
            if (supertrend_4h_dir_aligned[i] == -1 or not bullish_momentum or not volume_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h uptrend OR 1d bullish momentum OR no volume
            if (supertrend_4h_dir_aligned[i] == 1 or not bearish_momentum or not volume_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals