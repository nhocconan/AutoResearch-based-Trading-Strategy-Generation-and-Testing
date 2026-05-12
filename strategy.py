#!/usr/bin/env python3
# 12H_WILLIAMS_ALLIGATOR_1W_TREND_VOLUME_CONFIRMATION
# Hypothesis: Williams Alligator on weekly timeframe defines trend direction (jaw-teeth-lips alignment), 
# 12h price crosses the Alligator's teeth (8-period SMMA) with volume confirmation for entry.
# Exit when price crosses the Alligator's jaw (13-period SMMA). 
# Uses weekly trend filter to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "12H_WILLIAMS_ALLIGATOR_1W_TREND_VOLUME_CONFIRMATION"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if length < 1:
        return source.copy()
    smma = np.full_like(source, np.nan, dtype=np.float64)
    smma[length-1] = np.mean(source[:length])
    for i in range(length, len(source)):
        smma[i] = (smma[i-1] * (length-1) + source[i]) / length
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator from weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMMA of median price
    median_price = (df_1w['high'].values + df_1w['low'].values) / 2
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)   # Red line
    lips = smma(median_price, 5)    # Green line
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Volume spike detection (20-period volume MA)
    vol_ma = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(20-1, len(volume)):
        vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > vol_ma * 2.0  # Volume > 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above Teeth with volume spike, Alligator aligned bullish (Lips > Teeth > Jaw)
            if (close[i] > teeth_aligned[i] and vol_spike[i] and 
                lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below Teeth with volume spike, Alligator aligned bearish (Lips < Teeth < Jaw)
            elif (close[i] < teeth_aligned[i] and vol_spike[i] and 
                  lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Jaw
            if close[i] < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Jaw
            if close[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals