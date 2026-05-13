#!/usr/bin/env python3
# 4h_Donchian_Breakout_20_Trix_50_Volume_Spike
# Hypothesis: On 4h timeframe, breakout beyond Donchian(20) channels with TRIX(50) momentum confirmation and volume spike captures sustained momentum moves in both bull and bear markets. The TRIX filter reduces whipsaws by ensuring momentum alignment, while volume confirms institutional participation. Designed for low-frequency, high-quality setups.

name = "4h_Donchian_Breakout_20_Trix_50_Volume_Spike"
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

    # Get 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Calculate TRIX(50) on daily close
    # TRIX = EMA(EMA(EMA(close, 50), 50), 50) - 1
    ema1 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema2 = pd.Series(ema1).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema3 = pd.Series(ema2).ewm(span=50, adjust=False, min_periods=50).mean().values
    trix_raw = (ema3 / np.roll(ema3, 1)) - 1  # % change
    trix_raw[0] = 0  # first value has no previous
    trix = trix_raw  # already in percent form

    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)

    # Calculate Donchian channels (20-period) on 4h data
    # Upper = max(high, 20), Lower = min(low, 20)
    df = pd.DataFrame({'high': high, 'low': low})
    donch_high = df['high'].rolling(window=20, min_periods=20).max().values
    donch_low = df['low'].rolling(window=20, min_periods=20).min().values

    # Volume spike: volume > 2.0 * 20-period average (~3.3 days at 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend momentum + breakout above Donchian high + volume spike
            if trix_aligned[i] > 0 and close[i] > donch_high[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend momentum + breakdown below Donchian low + volume spike
            elif trix_aligned[i] < 0 and close[i] < donch_low[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or momentum turns negative
            if close[i] < donch_low[i] or trix_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or momentum turns positive
            if close[i] > donch_high[i] or trix_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals