#!/usr/bin/env python3
"""
4h_Donchian_Breakout_With_12hTrend_Volume
Hypothesis: Donchian(20) breakout on 4h with 12h EMA50 trend filter and volume confirmation.
Works in bull markets via breakout momentum and in bear markets via short breakdowns.
Volume confirmation filters false breakouts. Trend filter ensures alignment with higher timeframe.
Designed for 20-40 trades/year on 4h timeframe.
"""

name = "4h_Donchian_Breakout_With_12hTrend_Volume"
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

    # Get 12h data (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)

    close_12h = df_12h['close'].values

    # Calculate Donchian channels (20-period) on 4h
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        donchian_high = high_max_20[i]
        donchian_low = low_min_20[i]
        ema50_val = ema50_12h_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(donchian_high) or np.isnan(donchian_low) or np.isnan(ema50_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian high + uptrend + volume confirmation
            if close[i] > donchian_high and close[i] > ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian low + downtrend + volume confirmation
            elif close[i] < donchian_low and close[i] < ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below Donchian low or below 12h EMA50
            if close[i] < donchian_low or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above Donchian high or above 12h EMA50
            if close[i] > donchian_high or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals