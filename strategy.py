#!/usr/bin/env python3
"""
4h_WilliamsAlligator_TopBottom_Filter
Hypothesis: Williams Alligator identifies market direction via jaw/teeth/lips alignment. 
Enters long when lips cross above teeth AND price > 8-period EMA (bullish alignment).
Enters short when lips cross below teeth AND price < 8-period EMA (bearish alignment).
Uses 1-day Williams Fractals to filter counter-trend trades: only long above recent bullish fractal,
short below recent bearish fractal. Designed for low trade frequency (20-40/year) to minimize fee drag.
"""

name = "4h_WilliamsAlligator_TopBottom_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def calculate_alligator(close):
    """Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMMA"""
    # Smoothed Moving Average (SMMA) approximation using EMA with alpha=1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 1.0 / period
        res = np.zeros_like(arr)
        res[0] = arr[0]
        for i in range(1, len(arr)):
            res[i] = alpha * arr[i] + (1 - alpha) * res[i-1]
        return res
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)

    # Calculate Williams Fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Need 2-bar confirmation for fractals (Williams requires 2 bars after close)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )

    # Calculate Alligator on 4h data
    jaw, teeth, lips = calculate_alligator(close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after Alligator warmup (max period 13+8+5+8 for safety)
        # Get aligned values for current 4h bar
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        close_val = close[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        
        # Skip if any required data is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or
            np.isnan(bearish_fractal_val) or np.isnan(bullish_fractal_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Lips above Teeth (bullish alignment) AND price above recent bullish fractal
            if (lips_val > teeth_val and 
                close_val > bullish_fractal_val):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips below Teeth (bearish alignment) AND price below recent bearish fractal
            elif (lips_val < teeth_val and 
                  close_val < bearish_fractal_val):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Lips cross below Teeth OR price drops below bullish fractal
            if (lips_val < teeth_val or close_val < bullish_fractal_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Lips cross above Teeth OR price rises above bearish fractal
            if (lips_val > teeth_val or close_val > bearish_fractal_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals