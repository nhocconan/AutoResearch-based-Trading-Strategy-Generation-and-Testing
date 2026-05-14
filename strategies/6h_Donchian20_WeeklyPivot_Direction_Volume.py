#!/usr/bin/env python3
# 6h_Donchian20_WeeklyPivot_Direction_Volume
# Hypothesis: Use 6h Donchian(20) breakout with weekly pivot direction filter (1w pivot > price = long bias, < price = short bias) and volume confirmation.
# Weekly pivot provides structural bias from higher timeframe, reducing false breakouts in sideways markets.
# Works in bull (follows breakouts with bullish weekly bias) and bear (avoids bullish breakouts in bearish weekly bias).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for weekly pivot calculation (using 12h to approximate weekly structure)
    df_12h = get_htf_data(prices, '12h')
    # Calculate weekly pivot points from 12h data (approximating weekly from 5x 12h periods)
    # Use typical price for pivot calculation
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    # Weekly pivot: using last 5 periods of 12h data (5 * 12h = 60h ≈ weekly)
    lookback = min(5, len(typical_price_12h))
    if lookback >= 3:
        # Calculate pivot from recent 12h data
        recent_high = df_12h['high'].rolling(window=lookback, min_periods=lookback).max().values
        recent_low = df_12h['low'].rolling(window=lookback, min_periods=lookback).min().values
        recent_close = df_12h['close'].rolling(window=lookback, min_periods=lookback).mean().values
        pivot_12h = (recent_high + recent_low + recent_close) / 3.0
    else:
        pivot_12h = typical_price_12h.rolling(window=5, min_periods=1).mean().values
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)

    # Get 6h data for Donchian channel
    df_6h = get_htf_data(prices, '6h')
    donchian_high = df_6h['high'].rolling(window=20, min_periods=20).max().values
    donchian_low = df_6h['low'].rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(pivot_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Donchian high + price above weekly pivot (bullish bias) + volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > pivot_12h_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low + price below weekly pivot (bearish bias) + volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < pivot_12h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low or weekly pivot turns bearish
            if (close[i] < donchian_low_aligned[i] or close[i] < pivot_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high or weekly pivot turns bullish
            if (close[i] > donchian_high_aligned[i] or close[i] > pivot_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals