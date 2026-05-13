#!/usr/bin/env python3
# 6h_FisherTransform_Trend_Reversal_v1
# Hypothesis: The Ehlers Fisher Transform identifies turning points in price cycles.
# On 6h timeframe, we use Fisher(10) crossing above -1.5 for longs and below +1.5 for shorts,
# filtered by daily trend (price vs EMA50) and volume spikes to avoid false signals in chop.
# This mean-reversion mechanism works in both bull (buying dips) and bear (selling rallies) markets.
# Targets 10-25 trades/year by requiring confluence of Fisher extreme, trend alignment, and volume.

name = "6h_FisherTransform_Trend_Reversal_v1"
timeframe = "6h"
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

    # === 1. Fisher Transform (10-period) on close ===
    # Normalize price to [-1, 1] over lookback period
    def fish_transform(series, length):
        if len(series) < length:
            return np.full_like(series, np.nan)
        highest = pd.Series(series).rolling(window=length, min_periods=length).max().values
        lowest = pd.Series(series).rolling(window=length, min_periods=length).min().values
        # Avoid division by zero
        diff = highest - lowest
        diff[diff == 0] = 1e-10
        # Normalize to [-1, 1]
        value = 2 * ((series - lowest) / diff - 0.5)
        # Clamp to avoid math domain errors
        value = np.clip(value, -0.999, 0.999)
        # Fisher transform
        fish = 0.5 * np.log((1 + value) / (1 - value))
        # Smoothed
        fish_smoothed = pd.Series(fish).ewm(alpha=0.5, adjust=False).mean().values
        return fish_smoothed

    fish = fish_transform(close, 10)

    # === 2. Daily trend filter: EMA50 ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # === 3. Volume spike: > 2.0 x 20-period average ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(fish[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Fisher crosses above -1.5 (oversold reversal) + uptrend + volume spike
            if fish[i] > -1.5 and fish[i-1] <= -1.5 and close[i] > ema50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Fisher crosses below +1.5 (overbought reversal) + downtrend + volume spike
            elif fish[i] < 1.5 and fish[i-1] >= 1.5 and close[i] < ema50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fisher crosses below +1.5 (overbought) or trend turns bearish
            if fish[i] < 1.5 and fish[i-1] >= 1.5 or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fisher crosses above -1.5 (oversold) or trend turns bullish
            if fish[i] > -1.5 and fish[i-1] <= -1.5 or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals