#!/usr/bin/env python3
"""
1D_Alligator_Fisher_Breakout
Hypothesis: Williams Alligator (JAWS/TEETH/LIPS) defines trend direction, Fisher Transform catches reversals within that trend, and volume filter confirms momentum. Works in bull/bear by following Alligator alignment.
"""

name = "1D_Alligator_Fisher_Breakout"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Williams Alligator parameters
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2
    
    # Alligator lines (smoothed median price)
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean()
    jaw = jaw.shift(jaw_shift)
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean()
    teeth = teeth.shift(teeth_shift)
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean()
    lips = lips.shift(lips_shift)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # Align Alligator lines to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_values)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_values)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_values)
    
    # Fisher Transform on daily close (9-period)
    price = close
    # Normalize price to [-1, 1] range over 9 periods
    max_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    min_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    range_hl = max_high - min_low
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1, range_hl)
    # Normalize to [-1, 1]
    value = 2 * ((price - min_low) / range_hl - 0.5)
    # Clamp to prevent math domain error
    value = np.clip(value, -0.999, 0.999)
    # Fisher Transform
    fisher = 0.5 * np.log((1 + value) / (1 - value))
    # Signal line
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = np.nan
    
    # Align Fisher Transform to daily (already on daily, but ensure alignment)
    fisher_aligned = align_htf_to_ltf(prices, prices, fisher)  # Self-align for daily
    fisher_signal_aligned = align_htf_to_ltf(prices, prices, fisher_signal)
    
    # Volume filter: above 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup periods
    start_idx = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 9, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(fisher_aligned[i]) or np.isnan(fisher_signal_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Alligator bullish (JAW > TEETH > LIPS) + Fisher crosses above signal + volume
            if (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and
                fisher_aligned[i] > fisher_signal_aligned[i] and
                fisher_aligned[i-1] <= fisher_signal_aligned[i-1] and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator bearish (JAW < TEETH < LIPS) + Fisher crosses below signal + volume
            elif (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and
                  fisher_aligned[i] < fisher_signal_aligned[i] and
                  fisher_aligned[i-1] >= fisher_signal_aligned[i-1] and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns bearish OR Fisher crosses below signal
            if (jaw_aligned[i] <= teeth_aligned[i] or teeth_aligned[i] <= lips_aligned[i] or
                fisher_aligned[i] < fisher_signal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns bullish OR Fisher crosses above signal
            if (jaw_aligned[i] >= teeth_aligned[i] or teeth_aligned[i] >= lips_aligned[i] or
                fisher_aligned[i] > fisher_signal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals