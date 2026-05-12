#!/usr/bin/env python3
"""
12h_1w_Alligator_Trend_Filter_Volume
Hypothesis: Williams Alligator on weekly timeframe identifies strong trends.
In bull markets, price stays above Alligator teeth (green line); in bear, below.
Price crossing the Alligator jaw (blue line) with volume confirmation signals trend changes.
Uses 12h timeframe for entries with 1w Alligator filter to reduce whipsaws.
Target: 15-35 trades/year per symbol.
"""

name = "12h_1w_Alligator_Trend_Filter_Volume"
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

    # Get weekly data for Alligator (13,8,5 SMAs shifted)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values

    # Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(high_1w).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(low_1w).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)

    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values

    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_vals)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above Alligator Jaw (bullish alignment) + volume spike
            # Bullish alignment: Lips > Teeth > Jaw
            if (close[i] > jaw_aligned[i] and 
                lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below Alligator Jaw (bearish alignment) + volume spike
            # Bearish alignment: Lips < Teeth < Jaw
            elif (close[i] < jaw_aligned[i] and 
                  lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Alligator Teeth or bearish alignment
            if (close[i] < teeth_aligned[i] or 
                lips_aligned[i] < teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Alligator Teeth or bullish alignment
            if (close[i] > teeth_aligned[i] or 
                lips_aligned[i] > teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals