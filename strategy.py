#!/usr/bin/env python3
"""
4h_Williams_Alligator_Filtered_v1
Hypothesis: Williams Alligator (3 SMAs) identifies trend direction; enter on pullback to median SMA with volume confirmation in 4h timeframe.
- Jaw (13-period), Teeth (8-period), Lips (5-period) SMAs on median price
- Bullish when Lips > Teeth > Jaw; Bearish when Lips < Teeth < Jaw
- Entry on pullback to Teeth (8 SMA) with volume > 1.5x average volume
- Exit on opposite Alligator alignment or volume drop
- Williams Alligator works in both trending and ranging markets by showing convergence/divergence
- Target: 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Williams_Alligator_Filtered_v1"
timeframe = "4h"
leverage = 1.0

def calculate_sma(arr, period):
    """Calculate simple moving average with proper handling of NaN"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    
    sma = np.full_like(arr, np.nan, dtype=float)
    for i in range(period - 1, len(arr)):
        sma[i] = np.mean(arr[i - period + 1:i + 1])
    return sma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price (typical price)
    median_price = (high + low + close) / 3.0
    
    # Williams Alligator components: Smoothed SMAs of median price
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = calculate_sma(median_price, jaw_period)
    teeth = calculate_sma(median_price, teeth_period)
    lips = calculate_sma(median_price, lips_period)
    
    # Alligator alignment: Bullish when Lips > Teeth > Jaw, Bearish when Lips < Teeth < Jaw
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Volume confirmation: current volume > 1.5x average volume over 20 periods
    vol_ma = calculate_sma(volume, 20)
    volume_confirmation = volume > (vol_ma * 1.5)
    
    # Pullback to Teeth (8 SMA) with tolerance
    pullback_tolerance = 0.001  # 0.1% tolerance
    lips_above_teeth = lips > teeth * (1 - pullback_tolerance)
    lips_below_teeth = lips < teeth * (1 + pullback_tolerance)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = max(jaw_period, teeth_period, lips_period, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: bearish alignment or volume drops below average
            if bearish_alignment[i] or volume[i] <= vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: bullish alignment or volume drops below average
            if bullish_alignment[i] or volume[i] <= vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: bullish alignment + pullback to teeth + volume confirmation
            if (bullish_alignment[i] and 
                lips_below_teeth[i] and 
                volume_confirmation[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish alignment + pullback to teeth + volume confirmation
            elif (bearish_alignment[i] and 
                  lips_above_teeth[i] and 
                  volume_confirmation[i]):
                position = -1
                signals[i] = -0.25
    
    return signals