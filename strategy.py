#!/usr/bin/env python3
# 6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation captures momentum in both bull and bear markets. Weekly pivot direction avoids counter-trend trades, volume confirms strength, and Donchian breakout provides clear entry/exit.

name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
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

    # ATR for volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Weekly pivot direction (load once, align)
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot_dir = np.where(weekly_close > weekly_pivot, 1, -1)  # 1 = bullish bias, -1 = bearish bias
    weekly_pivot_dir_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_dir)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_pivot_dir_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + weekly pivot bullish + volume spike
            if (close[i] > donchian_high[i] and 
                weekly_pivot_dir_aligned[i] == 1 and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + weekly pivot bearish + volume spike
            elif (close[i] < donchian_low[i] and 
                  weekly_pivot_dir_aligned[i] == -1 and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or volume drops
            if close[i] < donchian_low[i] or volume[i] < vol_avg_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or volume drops
            if close[i] > donchian_high[i] or volume[i] < vol_avg_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals