#!/usr/bin/env python3
# 12h_Vortex_Trend_With_WeeklyTrend_Filter
# Hypothesis: The Vortex indicator identifies trend direction (VI+ > VI- for uptrend, VI- > VI+ for downtrend).
# Trades only in the direction of the weekly trend (price above/below weekly EMA200) to avoid counter-trend whipsaws.
# Uses volume confirmation (volume > 1.5x 20-period average) to filter low-quality breakouts.
# Designed for low-frequency, high-quality signals on 12h timeframe to minimize fee drag and work in both bull and bear markets.

name = "12h_Vortex_Trend_With_WeeklyTrend_Filter"
timeframe = "12h"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values

    # Weekly EMA200 for trend filter (needs minimum period)
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # Calculate Vortex Indicator (VI) on 12h data
    # True Range (TR)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close

    # Positive and Negative Vortex Movement
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0

    # Sum over 14 periods (standard Vortex period)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values

    # VI+ and VI-
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14

    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(vi_plus[i]) or 
            np.isnan(vi_minus[i]) or
            np.isnan(ema200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ > VI- (uptrend) + price above weekly EMA200 + volume spike
            if vi_plus[i] > vi_minus[i] and close[i] > ema200_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (downtrend) + price below weekly EMA200 + volume spike
            elif vi_minus[i] > vi_plus[i] and close[i] < ema200_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend turns down (VI- > VI+) or price crosses below weekly EMA200
            if vi_minus[i] > vi_plus[i] or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend turns up (VI+ > VI-) or price crosses above weekly EMA200
            if vi_plus[i] > vi_minus[i] or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals