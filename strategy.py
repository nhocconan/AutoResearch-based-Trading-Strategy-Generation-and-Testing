#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Rebound_Volume_Confirmation_v1
Hypothesis: Price often reverses from daily Camarilla S3/R3 levels rather than breaking through.
We take counter-trend positions at these strong support/resistance zones with volume confirmation.
Works in both bull (buy dips at S3) and bear (sell rallies at R3) markets by fading extremes.
Target: 15-25 trades/year per symbol to minimize fee drag.
"""

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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's close for calculations
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # first bar uses its own close
    
    range_1d = high_1d - low_1d
    
    # Resistance levels (R3)
    R3 = close_prev + (range_1d * 1.2500 / 4)
    
    # Support levels (S3)
    S3 = close_prev - (range_1d * 1.2500 / 4)
    
    # Align levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume filter: current volume > 1.3x 20-period average (less strict than breakout)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_confirmation = volume > (vol_ma_20 * 1.3)
    
    # Price proximity: within 0.5% of S3/R3 level
    proximity_to_S3 = np.abs(close - S3_aligned) / S3_aligned < 0.005
    proximity_to_R3 = np.abs(close - R3_aligned) / R3_aligned < 0.005
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(volume_confirmation[i]) or
            np.isnan(proximity_to_S3[i]) or np.isnan(proximity_to_R3[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: price near S3 with volume confirmation (buy the dip)
        long_setup = proximity_to_S3[i] and volume_confirmation[i]
        
        # Short setup: price near R3 with volume confirmation (sell the rally)
        short_setup = proximity_to_R3[i] and volume_confirmation[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = position_size
        elif short_setup and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Camarilla_Pivot_Rebound_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0