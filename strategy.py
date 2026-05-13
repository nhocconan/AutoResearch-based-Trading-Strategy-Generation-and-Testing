#!/usr/bin/env python3
# 1h_4h_Donchian_Breakout_1dTrend_VolumeSpike
# Hypothesis: On 1h timeframe, breakout beyond 4h Donchian channels (20-period) with alignment to daily trend (price vs EMA50) and volume confirmation captures momentum moves while minimizing false signals.
# Uses 4h for trend/structure and 1d for higher timeframe trend filter. Volume spike filters for conviction.
# Designed for low-frequency, high-quality setups to avoid fee drag on 1h chart.

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
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values

    # Align Donchian channels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume spike: volume > 2.0 * 24-period average (~1 day at 1h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + breakout above 4h Donchian high + volume spike
            if close[i] > ema50_aligned[i] and close[i] > donchian_high_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Downtrend + breakdown below 4h Donchian low + volume spike
            elif close[i] < ema50_aligned[i] and close[i] < donchian_low_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 4h Donchian low or trend turns bearish
            if close[i] < donchian_low_aligned[i] or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above 4h Donchian high or trend turns bullish
            if close[i] > donchian_high_aligned[i] or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals