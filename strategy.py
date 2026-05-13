#!/usr/bin/env python3
# 4h_ADX_Trend_Filter_Breakout
# Hypothesis: Trade breakouts of 4h 20-period high/low only when 1d ADX > 25 (strong trend).
# Use 1d ADX as trend filter to avoid whipsaws in ranging markets.
# Breakout logic: long when price > highest(high,20), short when price < lowest(low,20).
# Position size: 0.25. Exit when opposite breakout occurs or ADX drops below 20.
# Designed to capture strong trends in both bull and bear markets while avoiding chop.
# Target: 20-40 trades/year per symbol to minimize fee drag.

name = "4h_ADX_Trend_Filter_Breakout"
timeframe = "4h"
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

    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]

    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # positive when down
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values

    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values

    # Handle division by zero
    adx = np.where((plus_di + minus_di) == 0, 0, adx)

    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above 20-period high in strong uptrend (ADX > 25)
            if close[i] > highest_high[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below 20-period low in strong downtrend (ADX > 25)
            elif close[i] < lowest_low[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below 20-period low OR trend weakens (ADX < 20)
            if close[i] < lowest_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above 20-period high OR trend weakens (ADX < 20)
            if close[i] > highest_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals