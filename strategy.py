#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Trend_Confirmation
Hypothesis: Williams Alligator (3 SMAs) confirms daily trend direction. Price above/below Alligator teeth with volume confirmation. Works in bull/bear via trend filter + volume filter to avoid whipsaws.
"""

name = "1d_WilliamsAlligator_Trend_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values

    # Align to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)

    # Volume confirmation: 1.5x 20-day SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above teeth and lips > jaw (bullish alignment) + volume spike
            if (close[i] > teeth_aligned[i] and 
                lips_aligned[i] > jaw_aligned[i] and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below teeth and lips < jaw (bearish alignment) + volume spike
            elif (close[i] < teeth_aligned[i] and 
                  lips_aligned[i] < jaw_aligned[i] and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below teeth
            if close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above teeth
            if close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals