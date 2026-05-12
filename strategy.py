#!/usr/bin/env python3
"""
6h_1D_WilliamsAlligator_Trend_With_VolumeFilter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) from daily timeframe provides trend direction and market phase awareness (sleeping/awake/feeding). Combined with 60-period volume moving average filter to ensure participation during active market phases. Works in both bull and bear markets by following the higher-timeframe trend while avoiding whipsaws during low-volume consolidation. Target: 20-50 trades/year per symbol.
"""

name = "6h_1D_WilliamsAlligator_Trend_With_VolumeFilter"
timeframe = "6h"
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

    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Williams Alligator (SMMA based)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price_1d = (high_1d + low_1d) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average (similar to Wilder's smoothing)"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result

    jaw_1d = smma(median_price_1d, 13)
    teeth_1d = smma(median_price_1d, 8)
    lips_1d = smma(median_price_1d, 5)

    # Shift as per Alligator definition
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    
    # Set NaN for shifted values that don't have data
    jaw_1d[:8] = np.nan
    teeth_1d[:5] = np.nan
    lips_1d[:3] = np.nan

    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)

    # Volume filter: 60-period volume moving average
    volume_series = pd.Series(volume)
    volume_ma60 = volume_series.rolling(window=60, min_periods=60).mean().values
    volume_threshold = volume_ma60 * 1.2  # Require 20% above average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after volume MA warmup
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_ma60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Alligator alignment checks
        # Bullish alignment: Lips > Teeth > Jaw (all lines ascending and separated)
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        # Bearish alignment: Jaw > Teeth > Lips (all lines descending and separated)
        bearish_aligned = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]

        if position == 0:
            # LONG: Bullish alignment with volume confirmation
            if bullish_aligned and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish alignment with volume confirmation
            elif bearish_aligned and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: When Alligator lines intertwine (market sleeping) or bearish alignment
            if not bullish_aligned:  # Exit when not in perfect bullish alignment
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: When Alligator lines intertwine or bullish alignment
            if not bearish_aligned:  # Exit when not in perfect bearish alignment
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals