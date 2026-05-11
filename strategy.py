#!/usr/bin/env python3
"""
6h_ElderRay_Alligator_Combo_v1
Hypothesis: Combine Elder Ray (bull/bear power) with Williams Alligator (jaw/teeth/lips) on 1d timeframe.
Elder Ray measures bull/bear power relative to EMA13; Alligator shows trend alignment.
In bull markets: strong bull power + Alligator aligned bullish (lips>teeth>jaw).
In bear markets: strong bear power + Alligator aligned bearish (jaw>teeth>lips).
This dual-filter reduces whipsaws and works in both regimes.
Target: 50-150 total trades over 4 years on 6h timeframe.
"""

name = "6h_ElderRay_Alligator_Combo_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1D Data for Elder Ray and Alligator ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:  # Need enough data for EMA13 and Alligator
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 26
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Elder Ray signals
        strong_bull_power = bull_power_aligned[i] > 0
        strong_bear_power = bear_power_aligned[i] < 0
        
        # Alligator alignment
        alligator_bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_bearish = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        
        if position == 0:
            # Long: strong bull power + Alligator bullish alignment
            if strong_bull_power and alligator_bullish:
                signals[i] = 0.25
                position = 1
            # Short: strong bear power + Alligator bearish alignment
            elif strong_bear_power and alligator_bearish:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weak bull power OR Alligator not bullish
            if not strong_bull_power or not alligator_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: weak bear power OR Alligator not bearish
            if not strong_bear_power or not alligator_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals