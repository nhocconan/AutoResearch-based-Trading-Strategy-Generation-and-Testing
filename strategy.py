#!/usr/bin/env python3
# 1d_1W_Donchian_Breakout_WeeklyTrend_Volume
# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
# Works in bull markets by catching breakouts, and in bear markets by requiring weekly trend alignment
# and volume confirmation to avoid false signals. Targets 15-25 trades/year on 1d timeframe.

name = "1d_1W_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate daily Donchian channels (20-period)
    # Highest high of last 20 days (including current)
    high_series = pd.Series(high)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 days (including current)
    low_series = pd.Series(low)
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 1.5x average of last 20 days
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter
        bullish_weekly = close[i] > ema_1w_aligned[i]
        bearish_weekly = close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: Break above Donchian high with bullish weekly trend and volume
            if (close[i] > donchian_high[i] and bullish_weekly and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low with bearish weekly trend and volume
            elif (close[i] < donchian_low[i] and bearish_weekly and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Donchian low or weekly trend turns bearish
            if close[i] < donchian_low[i] or not bullish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Donchian high or weekly trend turns bullish
            if close[i] > donchian_high[i] or not bearish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals