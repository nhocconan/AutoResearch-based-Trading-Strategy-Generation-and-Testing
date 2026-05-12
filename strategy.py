#!/usr/bin/env python3

# 1d_1W_12H_EMA_Crossover_Volume_Momentum
# Hypothesis: Use weekly EMA trend filter with daily EMA crossover (21/55) for momentum,
# combined with volume confirmation. Weekly trend ensures alignment with higher timeframe
# momentum, while EMA crossover captures medium-term trends. Volume filters weak moves.
# Designed for low frequency (15-25 trades/year) to work in both bull and bear markets
# by following the dominant trend on multiple timeframes.

name = "1d_1W_12H_EMA_Crossover_Volume_Momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Get 12-hour data for faster trend confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:
        return np.zeros(n)

    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    # Daily EMA crossover (21/55) for momentum
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values

    # Volume confirmation: current volume > 1.5x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(55, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(ema21[i]) or np.isnan(ema55[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Multi-timeframe trend alignment
        weekly_bullish = close[i] > ema_1w_aligned[i]
        weekly_bearish = close[i] < ema_1w_aligned[i]
        twelve_hour_bullish = close[i] > ema_12h_aligned[i]
        twelve_hour_bearish = close[i] < ema_12h_aligned[i]

        # EMA crossover signals
        ema_bullish_cross = ema21[i] > ema55[i] and ema21[i-1] <= ema55[i-1]
        ema_bearish_cross = ema21[i] < ema55[i] and ema21[i-1] >= ema55[i-1]

        if position == 0:
            # LONG: Bullish alignment across timeframes + EMA bullish cross + volume
            if (weekly_bullish and twelve_hour_bullish and ema_bullish_cross and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish alignment across timeframes + EMA bearish cross + volume
            elif (weekly_bearish and twelve_hour_bearish and ema_bearish_cross and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Breakdown in alignment or EMA bearish cross
            if (not weekly_bullish or not twelve_hour_bullish or ema_bearish_cross):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Breakdown in alignment or EMA bullish cross
            if (not weekly_bearish or not twelve_hour_bearish or ema_bullish_cross):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals