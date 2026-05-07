#!/usr/bin/env python3
"""
12h_Alligator_Top_Bottom_Reversal
Hypothesis: Williams Alligator identifies trends on 1d chart. When price crosses below Alligator's teeth (red line) in a downtrend,
it signals a potential short-term top for mean-reversion short. When price crosses above teeth in an uptrend,
it signals a potential short-term bottom for mean-reversion long. Uses 12h chart for entry timing with volume confirmation.
Designed for low trade frequency (~15-30/year) to minimize fee drag and work in both bull and bear markets by fading
short-term extremes within the longer-term trend context.
"""
timeframe = "12h"
name = "12h_Alligator_Top_Bottom_Reversal"
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
    
    # 1d data for Alligator indicator (Williams Alligator: Jaw=13, Teeth=8, Lips=5, all shifted)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Alligator components (smoothed medians with shift)
    close_1d = df_1d['close'].values
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth: 8-period SMMA shifted 5 bars  
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips: 5-period SMMA shifted 3 bars
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_vals)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for warmup
        # Skip if any critical value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine trend direction: Lips above Teeth = uptrend, Lips below Teeth = downtrend
            is_uptrend = lips_aligned[i] > teeth_aligned[i]
            
            # Long: price crosses above Teeth in uptrend (potential bottom) + volume confirmation
            if close[i] > teeth_aligned[i] and close[i-1] <= teeth_aligned[i-1] and is_uptrend and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below Teeth in downtrend (potential top) + volume confirmation
            elif close[i] < teeth_aligned[i] and close[i-1] >= teeth_aligned[i-1] and not is_uptrend and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below Lips (end of short-term move) or trend changes
            if close[i] < lips_aligned[i] or lips_aligned[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above Lips (end of short-term move) or trend changes
            if close[i] > lips_aligned[i] or lips_aligned[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals