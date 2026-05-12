#!/usr/bin/env python3
# 6h_PivotPoint_Reversal_Strategy
# Hypothesis: 6h Pivot Point reversal strategy using 12h pivot levels for context and volume confirmation.
# Longs near S1/S2 in 12h uptrend, shorts near R1/R2 in 12h downtrend.
# Pivot points provide institutional support/resistance levels that work in both bull and bear markets.
# Targets 50-150 total trades over 4 years (12-37/year) with 0.25 position size to minimize fee drag.

name = "6h_PivotPoint_Reversal_Strategy"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 12h data for pivot point calculation (higher timeframe context)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)

    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate pivot points for 12h timeframe
    # Pivot Point = (High + Low + Close) / 3
    pp = (high_12h + low_12h + close_12h) / 3
    # Resistance 1 = (2 * PP) - Low
    r1 = (2 * pp) - low_12h
    # Support 1 = (2 * PP) - High
    s1 = (2 * pp) - high_12h
    # Resistance 2 = PP + (High - Low)
    r2 = pp + (high_12h - low_12h)
    # Support 2 = PP - (High - Low)
    s2 = pp - (high_12h - low_12h)

    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)

    # Calculate 6h volume SMA for confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.3  # Require 1.3x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price near support levels in 12h uptrend with volume confirmation
            # 12h uptrend: close > pivot point
            near_support = (close[i] <= s1_aligned[i] * 1.02) or (close[i] <= s2_aligned[i] * 1.02)
            in_uptrend = close_12h[-1] > pp[-1] if len(close_12h) > 0 and len(pp) > 0 else False
            # More robust uptrend check: current 12h close > current 12h pivot
            if i < len(pp_aligned):
                in_uptrend = close[i] > pp_aligned[i]  # Simplified: 6h close > 12h pivot
            
            if near_support and in_uptrend and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price near resistance levels in 12h downtrend with volume confirmation
            # 12h downtrend: close < pivot point
            near_resistance = (close[i] >= r1_aligned[i] * 0.98) or (close[i] >= r2_aligned[i] * 0.98)
            in_downtrend = close[i] < pp_aligned[i]  # 6h close < 12h pivot
            
            if near_resistance and in_downtrend and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot point or shows weakness
            if close[i] >= pp_aligned[i] or close[i] <= s1_aligned[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot point or shows strength
            if close[i] <= pp_aligned[i] or close[i] >= r1_aligned[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals