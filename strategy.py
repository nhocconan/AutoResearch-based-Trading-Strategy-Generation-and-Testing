#!/usr/bin/env python3
# 6h_12h_Donchian20_Breakout_Pullback
# Hypothesis: Enter on pullbacks to 20-period EMA after Donchian(20) breakouts on 6h timeframe, with 12h trend filter (EMA50) and volume confirmation. Designed for 50-150 total trades over 4 years to minimize fee drag while capturing trending moves in both bull and bear markets. Pullback entries improve win rate vs breakout-only strategies.

name = "6h_12h_Donchian20_Breakout_Pullback"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    # Calculate 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Calculate 6h EMA20 for pullback entries
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_20[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # 12h trend filter
        bullish_trend = close[i] > ema_12h_aligned[i]
        bearish_trend = close[i] < ema_12h_aligned[i]

        if position == 0:
            # LONG: Pullback to EMA20 after upward Donchian breakout with bullish 12h trend and volume
            if (close[i] > low_20[i] and close[i-1] <= low_20[i-1]) and \
               close[i] >= ema_20[i] * 0.995 and close[i] <= ema_20[i] * 1.005 and \
               bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Pullback to EMA20 after downward Donchian breakout with bearish 12h trend and volume
            elif (close[i] < high_20[i] and close[i-1] >= high_20[i-1]) and \
                 close[i] >= ema_20[i] * 0.995 and close[i] <= ema_20[i] * 1.005 and \
                 bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low or 12h trend turns bearish
            if close[i] < low_20[i] and close[i-1] >= low_20[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high or 12h trend turns bullish
            if close[i] > high_20[i] and close[i-1] <= high_20[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals