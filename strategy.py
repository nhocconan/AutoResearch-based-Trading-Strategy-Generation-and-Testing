#!/usr/bin/env python3
# 12h_WilliamsAlligator_Trend_Filter
# Hypothesis: Use Williams Alligator (13/8/5 SMAs) to identify trend direction on weekly timeframe.
# Enter long when price is above alligator teeth (middle SMA) and price > 1d close (trend alignment).
# Enter short when price is below alligator teeth and price < 1d close.
# Williams Alligator filters out sideways chop and captures trending moves.
# Works in bull (trend-following long) and bear (trend-following short).
# Low frequency due to weekly trend filter and strict price alignment.

name = "12h_WilliamsAlligator_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values

    # Get weekly data for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    close_1w = df_1w['close'].values
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values   # Red line
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values   # Green line
    
    # Align weekly indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(close_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > teeth (alligator bullish alignment) + price > 1d close
            if close[i] > teeth_aligned[i] and close[i] > close_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price < teeth (alligator bearish alignment) + price < 1d close
            elif close[i] < teeth_aligned[i] and close[i] < close_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < teeth (trend weakness) OR price < 1d close
            if close[i] < teeth_aligned[i] or close[i] < close_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > teeth (trend weakness) OR price > 1d close
            if close[i] > teeth_aligned[i] or close[i] > close_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals