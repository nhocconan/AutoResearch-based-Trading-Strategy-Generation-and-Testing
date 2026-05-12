#!/usr/bin/env python3
"""
6h_WilliamsAlligator_Trend_Confirmation_v3
Hypothesis: Williams Alligator (Jaws, Teeth, Lips) with MAs from 1d and 1w timeframes.
Long when Lips > Teeth > Jaws and price > Lips, short when Lips < Teeth < Jaws and price < Lips.
Exit when teeth and jaws cross (trend weakening). Uses 1d for Alligator lines, 1w for trend filter.
Targets 15-30 trades/year to stay within fee limits. Works in both bull (strong trends) and bear (clear downtrends).
"""

name = "6h_WilliamsAlligator_Trend_Confirmation_v3"
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

    # Get daily data for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate Alligator lines from daily close (SMMA equivalent using EMA as proxy)
    # Jaws: 13-period SMMA shifted 8 bars ahead -> EMA(13) shifted
    # Teeth: 8-period SMMA shifted 5 bars ahead -> EMA(8) shifted
    # Lips: 5-period SMMA shifted 3 bars ahead -> EMA(5) shifted
    jaws_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_1d = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_1d = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values

    # Shift to simulate SMMA with forward shift (Alligator's predictive nature)
    # In practice, we use the values as-is and wait for alignment
    jaws_1d = np.roll(jaws_1d, 8)  # shift 8 bars forward
    teeth_1d = np.roll(teeth_1d, 5)  # shift 5 bars forward
    lips_1d = np.roll(lips_1d, 3)   # shift 3 bars forward
    # Fill NaN from roll with first valid value
    jaws_1d[:8] = jaws_1d[8] if len(jaws_1d) > 8 else jaws_1d[-1]
    teeth_1d[:5] = teeth_1d[5] if len(teeth_1d) > 5 else teeth_1d[-1]
    lips_1d[:3] = lips_1d[3] if len(lips_1d) > 3 else lips_1d[-1]

    # Get weekly data for trend filter (only trade in direction of weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Align all HTF data to 6s timeframe
    jaws_1d_aligned = align_htf_to_ltf(prices, df_1d, jaws_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # warmup for indicators
        # Get aligned values for current bar
        jaws = jaws_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        weekly_ema = ema50_1w_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(jaws) or np.isnan(teeth) or np.isnan(lips) or 
            np.isnan(weekly_ema)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Lips > Teeth > Jaws (bullish alignment) AND price > Lips AND price > weekly EMA50
            if (lips > teeth > jaws and 
                close[i] > lips and 
                close[i] > weekly_ema):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaws (bearish alignment) AND price < Lips AND price < weekly EMA50
            elif (lips < teeth < jaws and 
                  close[i] < lips and 
                  close[i] < weekly_ema):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Teeth crosses below Jaws (trend weakening) OR price crosses below Teeth
            if (teeth < jaws) or (close[i] < teeth):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Teeth crosses above Jaws (trend weakening) OR price crosses above Teeth
            if (teeth > jaws) or (close[i] > teeth):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals