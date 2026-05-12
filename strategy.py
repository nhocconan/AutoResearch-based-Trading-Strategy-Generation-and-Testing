#!/usr/bin/env python3
"""
6h_ChandelierExit_Trend_With_1dVolumeFilter
Hypothesis: Chandelier Exit (ATR-based trailing stop) captures trends effectively in 6h timeframe.
Long when price crosses above Chandelier Long (high - ATR*3) with daily volume > 1.5x 20-day average.
Short when price crosses below Chandelier Short (low + ATR*3) with daily volume > 1.5x 20-day average.
Uses daily ADX > 25 to filter choppy regimes and ensure trending conditions.
Designed to work in both bull (capture uptrends) and bear (capture downtrends) markets by being directionally adaptive.
Targets 12-37 trades/year (50-150 total over 4 years) with low turnover to minimize fee drag.
"""

name = "6h_ChandelierExit_Trend_With_1dVolumeFilter"
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

    # Get daily data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate ATR(22) for Chandelier Exit (approx 6h periods in a day: 4, but use 22 for stability)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = high[0] - close[0]
    tr3[0] = close[0] - low[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=22, min_periods=22).mean().values

    # Chandelier Exit components
    chandelier_long = np.maximum.accumulate(high) - atr * 3.0  # long stop level
    chandelier_short = np.minimum.accumulate(low) + atr * 3.0  # short stop level

    # Daily volume average for filter
    vol_avg_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20d)

    # Daily ADX(14) for trend filter
    # +DM, -DM, TR
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # positive values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1)))
    )
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(22, n):
        # Skip if any required data is NaN
        if (np.isnan(chandelier_long[i]) or np.isnan(chandelier_short[i]) or
            np.isnan(vol_avg_20d_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: only trade when ADX > 25 (trending market)
        if adx_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above Chandelier Long + volume confirmation
            if (close[i] > chandelier_long[i] and
                volume[i] > vol_avg_20d_aligned[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below Chandelier Short + volume confirmation
            elif (close[i] < chandelier_short[i] and
                  volume[i] > vol_avg_20d_aligned[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Chandelier Long
            if close[i] < chandelier_long[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Chandelier Short
            if close[i] > chandelier_short[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals