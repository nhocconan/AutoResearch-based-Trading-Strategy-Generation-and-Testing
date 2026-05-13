#!/usr/bin/env python3
"""
6h_Keltner_Bollinger_Squeeze_With_Volume_Confirmation
Hypothesis: Combines Keltner Channel volatility breakout with Bollinger Band squeeze
to identify low-volatility breakouts in the direction of the 1d trend. Volume spike
confirms institutional participation. Designed to work in both bull and bear markets
by following the 1d trend direction. Targets low-frequency, high-quality setups
to minimize fee drift.
"""
name = "6h_Keltner_Bollinger_Squeeze_With_Volume_Confirmation"
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

    # Get daily data for trend filter and squeeze calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Bollinger Bands (20, 2) on 1d
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle

    # Keltner Channel (20, 1.5) on 1d
    kc_middle = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_1d = pd.Series(high_1d - low_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = kc_middle + 1.5 * atr_1d
    kc_lower = kc_middle - 1.5 * atr_1d

    # Squeeze condition: Bollinger Bands inside Keltner Channel
    squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)

    # Align to 6h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # 60-period (5-day) volume average for volume confirmation
    vol_ma_60 = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    volume_spike = volume > 2.0 * vol_ma_60

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma_60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LOCK: Bollinger Bands inside Keltner Channel (squeeze) + volume spike
            # Direction from 1d EMA50
            if squeeze_aligned[i] and volume_spike[i]:
                if close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Squeeze ends or trend turns bearish
            if not squeeze_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Squeeze ends or trend turns bullish
            if not squeeze_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals