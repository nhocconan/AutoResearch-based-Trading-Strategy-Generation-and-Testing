#!/usr/bin/env python3
"""
6h_1d_WilliamsAlligator_ElderRay_Breakout_v1
Concept: Combine Williams Alligator (trend filter) from 1d with Elder Ray (momentum) from 1d for 6h breakouts.
- Long when 1d Alligator is bullish (jaw < teeth < lips) AND Elder Ray bull power > 0 AND price breaks 6h high of 20 bars
- Short when 1d Alligator is bearish (jaw > teeth > lips) AND Elder Ray bear power < 0 AND price breaks 6h low of 20 bars
- Exit when Alligator alignment reverses or Elder Ray changes sign
- Williams Alligator: jaw=SMMA(13,8), teeth=SMMA(8,5), lips=SMMA(5,3)
- Elder Ray: bull power = high - EMA13, bear power = low - EMA13
- Designed to work in both bull (trend following) and bear (counter-trend via Alligator reversals)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsAlligator_ElderRay_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def smma(src, length, offset):
    """Smoothed Moving Average (SMMA) used in Williams Alligator"""
    if length <= 0:
        return src.copy()
    result = np.full_like(src, np.nan, dtype=float)
    # First value is simple SMA
    if len(src) >= length:
        result[length-1] = np.nansum(src[:length]) / length
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(src)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + src[i]) / length
        else:
            result[i] = src[i]
    # Apply offset (shift left by offset bars)
    if offset > 0:
        result = np.roll(result, -offset)
        result[-offset:] = np.nan
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Calculate 1d Williams Alligator ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # SMMA calculations
    jaw_1d = smma(close_1d, 13, 8)   # SMMA(13,8)
    teeth_1d = smma(close_1d, 8, 5)  # SMMA(8,5)
    lips_1d = smma(close_1d, 5, 3)   # SMMA(5,3)
    
    # === Calculate 1d Elder Ray (using EMA13) ===
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d   # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema13_1d    # Bear Power = Low - EMA13
    
    # Align Alligator components to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 6s: Calculate 20-period high/low for breakout ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Rolling max/min for 20-period breakout levels
    high_max20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators (20+13+offsets)
    
    for i in range(start_idx, n):
        # Get values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        high_max20_val = high_max20[i]
        low_min20_val = low_min20[i]
        close_val = close_6h[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or 
            np.isnan(bull_power_val) or np.isnan(bear_power_val) or
            np.isnan(high_max20_val) or np.isnan(low_min20_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment checks
        alligator_bullish = jaw_val < teeth_val < lips_val   # Jaw < Teeth < Lips
        alligator_bearish = jaw_val > teeth_val > lips_val   # Jaw > Teeth > Lips
        
        if position == 0:
            # Long: Bullish Alligator + positive Bull Power + breakout above 20-period high
            if alligator_bullish and bull_power_val > 0 and close_val > high_max20_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + negative Bear Power + breakdown below 20-period low
            elif alligator_bearish and bear_power_val < 0 and close_val < low_min20_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator turns bearish OR Bull Power becomes negative
            if not alligator_bullish or bull_power_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator turns bullish OR Bear Power becomes positive
            if not alligator_bearish or bear_power_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals