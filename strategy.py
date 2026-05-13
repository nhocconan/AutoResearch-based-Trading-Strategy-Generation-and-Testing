#!/usr/bin/env python3
# 12h_WilliamsAlligator_ElderRay_TripleFilter
# Hypothesis: Williams Alligator identifies trend direction (jaws/teeth/lips alignment),
# Elder Ray measures bull/bear power via EMA13, and volume spike confirms institutional participation.
# Combined, these filters provide high-probability trend-following entries in both bull and bear markets.
# Uses 1-day trend filter for higher timeframe alignment. Designed for low-frequency, high-quality setups
# to minimize fee drag on 12h timeframe. Target: 15-30 trades/year.

name = "12h_WilliamsAlligator_ElderRay_TripleFilter"
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

    # Get daily data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Williams Alligator: SMAs with specific periods
    # Jaw (13-period SMMA, shifted 8 bars)
    # Teeth (8-period SMMA, shifted 5 bars)
    # Lips (5-period SMMA, shifted 3 bars)
    # Using EMA as proxy for SMMA with same lookback for simplicity
    jaw = pd.Series(close_1d).ewm(span=13, adjust=False).mean().values
    teeth = pd.Series(close_1d).ewm(span=8, adjust=False).mean().values
    lips = pd.Series(close_1d).ewm(span=5, adjust=False).mean().values

    # Shift the averages (Alligator specific)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill initial NaNs from roll
    jaw_shifted[:8] = jaw[0] if not np.isnan(jaw[0]) else 0
    teeth_shifted[:5] = teeth[0] if not np.isnan(teeth[0]) else 0
    lips_shifted[:3] = lips[0] if not np.isnan(lips[0]) else 0

    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False).mean().values
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13

    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > 2.0 * vol_ma_20

    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))

    # 1-day EMA50 for trend filter (higher timeframe alignment)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Alligator aligned (Lips > Teeth > Jaw) + Bull Power > 0 + Volume spike + Price > EMA50
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                bull_power_aligned[i] > 0 and 
                volume_spike_aligned[i] > 0.5 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator aligned (Jaw > Teeth > Lips) + Bear Power < 0 + Volume spike + Price < EMA50
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                  bear_power_aligned[i] < 0 and 
                  volume_spike_aligned[i] > 0.5 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator reverses (Lips < Jaw) OR Bear Power < 0
            if lips_aligned[i] < jaw_aligned[i] or bear_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator reverses (Jaw < Lips) OR Bull Power > 0
            if jaw_aligned[i] < lips_aligned[i] or bull_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals