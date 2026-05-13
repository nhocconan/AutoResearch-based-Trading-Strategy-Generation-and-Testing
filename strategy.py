#!/usr/bin/env python3
# 4h_Donchian_Breakout_With_Volume_Confirmation
# Hypothesis: Donchian channel breakouts with volume confirmation and EMA trend filter
# capture institutional moves in both bull and bear markets. The 4h timeframe reduces
# trade frequency to manageable levels (< 50 trades/year), minimizing fee drag.
# Volume spike confirms institutional participation, while EMA200 ensures trend alignment.
# Exit on opposite Donchian breakout or trend reversal.

name = "4h_Donchian_Breakout_With_Volume_Confirmation"
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

    # Get daily data for Donchian calculation (more stable than 4h)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate 20-day Donchian channels
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll

    # Align to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)

    # 4h EMA200 for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values

    # Volume spike: volume > 2.0 * 20-period average (~5 days at 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema200[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + break above Donchian high + volume spike
            if close[i] > ema200[i] and close[i] > donchian_high_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + break below Donchian low + volume spike
            elif close[i] < ema200[i] and close[i] < donchian_low_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or trend turns bearish
            if close[i] < donchian_low_aligned[i] or close[i] < ema200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or trend turns bullish
            if close[i] > donchian_high_aligned[i] or close[i] > ema200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals