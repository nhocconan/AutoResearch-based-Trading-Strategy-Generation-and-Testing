#!/usr/bin/env python3
# 4h_Donchian20_PlusVolume_PlusTrend_v1
# Hypothesis: On 4h timeframe, Donchian(20) breakout with volume confirmation and trend filter (price vs EMA50)
# captures strong momentum moves. Trend filter uses price > EMA50 for longs, price < EMA50 for shorts.
# Volume spike (>2x 20-period MA) confirms institutional participation.
# Exit when price crosses back below/above EMA50 or Donchian opposite band.
# Designed for low-frequency, high-quality setups with clear risk control.

name = "4h_Donchian20_PlusVolume_PlusTrend_v1"
timeframe = "4h"
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

    # Get EMA50 for trend filter (using same timeframe for simplicity)
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values

    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + breakout above Donchian high + volume spike
            if close[i] > ema50[i] and close[i] > donchian_high[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + breakdown below Donchian low + volume spike
            elif close[i] < ema50[i] and close[i] < donchian_low[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA50 or Donchian low
            if close[i] < ema50[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA50 or Donchian high
            if close[i] > ema50[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals