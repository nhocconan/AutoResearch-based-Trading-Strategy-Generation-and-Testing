#!/usr/bin/env python3
# 6H_ElderRay_Alligator_TrendFilter
# Hypothesis: Combines Elder Ray (Bull/Bear Power) with Williams Alligator for trend confirmation on 6h timeframe.
# Elder Ray measures bull/bear power via EMA(13) divergence; Alligator (SMAs 13,8,5) filters trend direction.
# Works in both bull and bear markets by only taking trades in direction of Alligator alignment.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6H_ElderRay_Alligator_TrendFilter"
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
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (Bull/Bear Power)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume filter: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Alligator trend filter: 
        # Uptrend: Lips > Teeth > Jaw (all aligned upward)
        # Downtrend: Lips < Teeth < Jaw (all aligned downward)
        alligator_up = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_down = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Bull Power positive + Alligator uptrend + volume filter
            if (bull_power_aligned[i] > 0 and 
                alligator_up and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative + Alligator downtrend + volume filter
            elif (bear_power_aligned[i] < 0 and 
                  alligator_down and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative or Alligator loses alignment
            if bull_power_aligned[i] <= 0 or not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive or Alligator loses alignment
            if bear_power_aligned[i] >= 0 or not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals