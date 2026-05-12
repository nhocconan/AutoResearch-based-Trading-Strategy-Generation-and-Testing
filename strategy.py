#!/usr/bin/env python3
"""
4h_WilliamsAlligator_ElderRay_1wTrend
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and convergence/divergence, combined with Elder Ray (Bull/Bear Power) to measure trend strength, filtered by 1-week EMA50 trend. Works in bull markets (Alligator aligned up + positive Elder Ray) and bear markets (Alligator aligned down + negative Elder Ray). Uses 1-week trend filter to avoid counter-trend trades. Designed for low trade frequency (~20-40/year) to minimize fee drag.
"""

name = "4h_WilliamsAlligator_ElderRay_1wTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1-week data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # 1-week EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Williams Alligator: SMMA (Smoothed Moving Average) with periods 13,8,5 and offsets 8,5,3
    # Jaw (13-period, 8-bar offset): SMMA(close, 13) shifted 8 bars
    # Teeth (8-period, 5-bar offset): SMMA(close, 8) shifted 5 bars
    # Lips (5-period, 3-bar offset): SMMA(close, 5) shifted 3 bars
    # SMMA calculation: first value = SMA, then SMMA = (prev_SMMA*(period-1) + close) / period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        sma = np.full_like(arr, np.nan)
        sma[period-1] = np.mean(arr[:period])
        smma_vals = np.full_like(arr, np.nan)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(arr)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + arr[i]) / period
        return smma_vals

    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)

    # Apply Alligator offsets: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set first values to NaN due to roll
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan

    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Alligator aligned up (Lips > Teeth > Jaw) + Bull Power > 0 + 1w uptrend
            if lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i] and bull_power[i] > 0 and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator aligned down (Lips < Teeth < Jaw) + Bear Power < 0 + 1w downtrend
            elif lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i] and bear_power[i] < 0 and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks (Lips < Teeth) or 1w trend turns down
            if lips_shifted[i] < teeth_shifted[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks (Lips > Teeth) or 1w trend turns up
            if lips_shifted[i] > teeth_shifted[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals