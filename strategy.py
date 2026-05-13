#!/usr/bin/env python3
# 4h_Donchian_Breakout_12hTrend_Volume
# Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA50 trend filter and volume confirmation.
# Long when price breaks above upper band with 12h EMA uptrend and volume > 1.5x 24-period average.
# Short when price breaks below lower band with 12h EMA downtrend and volume spike.
# Exit when price returns to the middle band (mean reversion) or opposite band breakout.
# Designed for 4h timeframe to balance trade frequency and signal quality, targeting 20-50 trades/year.

name = "4h_Donchian_Breakout_12hTrend_Volume"
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
    volume = prices['volume'].values

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values

    # Calculate Donchian channels (20-period) on 4h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_channel = (high_max_20 + low_min_20) / 2.0

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: volume > 1.5x 24-period average
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above upper Donchian + 12h EMA uptrend + volume spike
            if (close[i] > high_max_20[i] and 
                close[i] > ema50_12h_aligned[i] and
                volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below lower Donchian + 12h EMA downtrend + volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < ema50_12h_aligned[i] and
                  volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below middle band (mean reversion)
            if close[i] < mid_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above middle band
            if close[i] > mid_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals