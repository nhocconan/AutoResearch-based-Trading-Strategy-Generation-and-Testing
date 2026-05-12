#!/usr/bin/env python3
"""
1d_Donchian_Breakout_WeeklyTrend_Volume
Hypothesis: Donchian channel breakouts on 1d with weekly trend filter and volume confirmation capture strong trending moves. Works in bull/bear by following higher timeframe (weekly) trend direction. Targets 20-50 trades over 4 years (5-12/year) to minimize fee drag.
"""

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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

    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')

    # Calculate weekly EMA50 for trend filter
    ema_50_weekly = pd.Series(df_weekly['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)

    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll

    # Volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(ema_50_weekly_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian upper + weekly uptrend + volume confirmation
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_weekly_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + weekly downtrend + volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_weekly_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian lower (breakdown)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian upper (breakout)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals