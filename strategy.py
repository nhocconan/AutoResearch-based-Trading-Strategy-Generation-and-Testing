#!/usr/bin/env python3
# 4h_Donchian_VolumeTrend_Pullback
# Hypothesis: Donchian(20) breakout in direction of 1d EMA50 trend with volume confirmation.
# Pullback to 20-period EMA for entry improves win rate and reduces false breakouts.
# Works in bull via breakout momentum, in bear via pullbacks during trend continuations.
# Target: 20-40 trades/year per symbol.

name = "4h_Donchian_VolumeTrend_Pullback"
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

    # Get 1d data for trend filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Donchian Channel (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # EMA20 for pullback entry
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema20[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Donchian high + 1d uptrend + volume spike + pullback to EMA20
            if close[i] > donchian_high[i] and close[i] > ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5 and close[i] <= ema20[i] * 1.01:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low + 1d downtrend + volume spike + pullback to EMA20
            elif close[i] < donchian_low[i] and close[i] < ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5 and close[i] >= ema20[i] * 0.99:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below Donchian low or 1d trend turns down
            if close[i] < donchian_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above Donchian high or 1d trend turns up
            if close[i] > donchian_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals