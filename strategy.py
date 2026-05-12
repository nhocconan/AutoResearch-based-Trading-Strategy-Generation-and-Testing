#!/usr/bin/env python3
"""
4h_Williams_Alligator_Trend_Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on 1d timeframe defines trend direction, with 4h price crossing above/below the Lips (fast line) as entry signal, confirmed by volume spike (>1.5x 20-period average). Works in bull/bear by following 1d trend direction, reducing false signals in ranging markets.
"""

name = "4h_Williams_Alligator_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator lines (SMMA-based)."""
    median_price = (high + low) / 2
    # SMMA (Smoothed Moving Average) calculation
    def smma(source, period):
        sma = np.full_like(source, np.nan, dtype=float)
        if len(source) >= period:
            sma[period-1] = np.mean(source[:period])
            for i in range(period, len(source)):
                sma[i] = (sma[i-1] * (period-1) + source[i]) / period
        return sma
    
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate Williams Alligator on 1d data
    jaw_1d, teeth_1d, lips_1d = williams_alligator(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        jaw_period=13,
        teeth_period=8,
        lips_period=5
    )

    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)

    # Volume spike: >1.5x 20-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # Start after Alligator warmup
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above Lips + Lips > Teeth > Jaw (uptrend) + volume spike
            if (close[i] > lips_aligned[i] and 
                lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below Lips + Lips < Teeth < Jaw (downtrend) + volume spike
            elif (close[i] < lips_aligned[i] and 
                  lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Lips (trend change)
            if close[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Lips (trend change)
            if close[i] > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals