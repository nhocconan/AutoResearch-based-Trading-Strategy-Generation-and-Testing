#!/usr/bin/env python3
# 6h_WilliamsAlligator_Trend_Filter
# Hypothesis: Williams Alligator (smoothed medians) combined with price position relative to the jaws.
# Long when price > Teeth and Jaws > Teeth (bullish alignment).
# Short when price < Teeth and Jaws < Teeth (bearish alignment).
# Uses 1d timeframe for Alligator calculation to avoid whipsaw, providing stable trend filtering.
# Works in bull markets by capturing trends and in bear markets by avoiding false signals during consolidation.

name = "6h_WilliamsAlligator_Trend_Filter"
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

    # Get 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')

    # Williams Alligator components on 1d: Smoothed medians (Jaw=13, Teeth=8, Lips=5)
    # Jaw: Smoothed median with period 13
    hlc_median_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    jaw = hlc_median_1d.rolling(window=13, min_periods=13).median().rolling(window=8, min_periods=8).mean().values
    # Teeth: Smoothed median with period 8
    teeth = hlc_median_1d.rolling(window=8, min_periods=8).median().rolling(window=5, min_periods=5).mean().values
    # Lips: Smoothed median with period 5
    lips = hlc_median_1d.rolling(window=5, min_periods=5).median().rolling(window=3, min_periods=3).mean().values

    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)

    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above Teeth and Jaws above Teeth (bullish alignment) with volume
            if (close[i] > teeth_aligned[i] and 
                jaw_aligned[i] > teeth_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below Teeth and Jaws below Teeth (bearish alignment) with volume
            elif (close[i] < teeth_aligned[i] and 
                  jaw_aligned[i] < teeth_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Teeth or alignment breaks
            if close[i] < teeth_aligned[i] or jaw_aligned[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Teeth or alignment breaks
            if close[i] > teeth_aligned[i] or jaw_aligned[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals