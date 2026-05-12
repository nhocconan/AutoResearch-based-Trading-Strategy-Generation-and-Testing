#!/usr/bin/env python3
# 12h_1D_1W_Donchian_Breakout_Volume_Trend
# Hypothesis: Donchian breakout on 12h with 1d volume confirmation and 1w trend filter.
# Enter long when price breaks above Donchian(20) high on 12h, volume > 1.5x 20-period average, and price > 1w EMA50.
# Enter short when price breaks below Donchian(20) low on 12h, volume > 1.5x 20-period average, and price < 1w EMA50.
# Exit when price crosses back through the Donchian midpoint or trend fails.
# Designed for low-frequency, high-conviction trades to minimize fee drag and work in both bull and bear markets.

name = "12h_1D_1W_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
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

    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate 20-period average volume on 1d
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)

    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate Donchian channels on 12h (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        vol_ok = volume[i] > 1.5 * vol_avg_20_aligned[i]

        if position == 0:
            # LONG: Donchian breakout up + volume confirmation + uptrend
            if (close[i] > donchian_high[i] and vol_ok and
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Donchian breakdown down + volume confirmation + downtrend
            elif (close[i] < donchian_low[i] and vol_ok and
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below Donchian midpoint OR trend fails
            if (close[i] < donchian_mid[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above Donchian midpoint OR trend fails
            if (close[i] > donchian_mid[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals