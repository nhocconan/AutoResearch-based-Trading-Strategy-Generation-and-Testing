#!/usr/bin/env python3
"""
12h_Vortex_VTIX_1wTrend
Hypothesis: Vortex Indicator (VI) identifies trend direction and strength, while VTIX (Vortex Trend Index) filters for strong trends. 
In trending markets (VTIX > 0.75), go long when VI+ > VI- and short when VI- > VI+. In ranging markets (VTIX < 0.25), 
fade extremes using 1-week Bollinger Bands. Uses volume confirmation to avoid weak signals.
"""

name = "12h_Vortex_VTIX_1wTrend"
timeframe = "12h"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values

    # Vortex Indicator (14-period)
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0  # First value undefined
    vm_minus[0] = 0

    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]

    # Sum over 14 periods
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values

    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum

    # VTIX (Vortex Trend Index) - measures trend strength
    vtix = np.abs(vi_plus - vi_minus) / (vi_plus + vi_minus)
    vtix = np.where((vi_plus + vi_minus) == 0, 0, vtix)

    # 1-week Bollinger Bands (20, 2)
    ma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    ma_20_aligned = align_htf_to_ltf(prices, df_1w, ma_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)

    # Volume confirmation: volume > 1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        if np.isnan(vtix[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or \
           np.isnan(ma_20_aligned[i]) or np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or \
           np.isnan(vol_avg_30[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Trending market: VTIX > 0.75
            if vtix[i] > 0.75:
                # LONG: VI+ > VI- and rising
                if vi_plus[i] > vi_minus[i] and vi_plus[i] > vi_plus[i-1] and volume[i] > vol_avg_30[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
                # SHORT: VI- > VI+ and rising
                elif vi_minus[i] > vi_plus[i] and vi_minus[i] > vi_minus[i-1] and volume[i] > vol_avg_30[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: VTIX < 0.25 - mean reversion at Bollinger Bands
            elif vtix[i] < 0.25:
                # LONG: price touches lower BB and shows rejection
                if close[i] <= lower_bb_aligned[i] and close[i] > close[i-1] and volume[i] > vol_avg_30[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
                # SHORT: price touches upper BB and shows rejection
                elif close[i] >= upper_bb_aligned[i] and close[i] < close[i-1] and volume[i] > vol_avg_30[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: trend weakens or reversal signals
            if vtix[i] < 0.5 or (vi_plus[i] < vi_minus[i] and vi_plus[i] < vi_plus[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: trend weakens or reversal signals
            if vtix[i] < 0.5 or (vi_minus[i] < vi_plus[i] and vi_minus[i] < vi_minus[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals