#!/usr/bin/env python3
# 1h_4h_Donchian_Breakout_1dTrend_VolumeSpike
# Hypothesis: On 1h timeframe, breakout beyond 4h Donchian channels (20-period) with alignment to 1d trend (price vs EMA50) and volume confirmation captures strong momentum moves.
# The 1d trend provides longer-term filter to reduce whipsaws, while 4h structure gives intermediate-term context.
# Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
# Designed for moderate-frequency, high-quality setups targeting 15-37 trades/year.

name = "1h_4h_Donchian_Breakout_1dTrend_VolumeSpike"
timeframe = "1h"
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

    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values

    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_4h, high_max_20)
    donchian_low = align_htf_to_ltf(prices, df_4h, low_min_20)

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume spike: volume > 2.0 * 24-period average (~1 day at 1h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24

    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + breakout above Donchian high + volume spike
            if close[i] > ema50_1d_aligned[i] and close[i] > donchian_high[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Downtrend + breakdown below Donchian low + volume spike
            elif close[i] < ema50_1d_aligned[i] and close[i] < donchian_low[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or trend turns bearish
            if close[i] < donchian_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or trend turns bullish
            if close[i] > donchian_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals