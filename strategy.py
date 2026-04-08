#!/usr/bin/env python3
# 4h_1d_1w_alligator_trend_v1
# Hypothesis: Use Williams Alligator on daily timeframe for trend direction, with price crossing the Alligator's teeth on 4h for entry, confirmed by volume surge.
# Long when price crosses above the Alligator's teeth (middle line) on 4h, with daily Alligator in bullish alignment (jaw < teeth < lips) and volume > 1.5x 20-period average.
# Short when price crosses below the Alligator's teeth on 4h, with daily Alligator in bearish alignment (jaw > teeth > lips) and volume surge.
# Exit when price crosses back below/above the teeth or when Alligator alignment changes.
# Designed for 4h timeframe to target 20-50 trades/year (80-200 total over 4 years).
# Williams Alligator identifies trends effectively, working in both bull and bear markets by filtering counter-trend noise.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_alligator_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Alligator
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Alligator lines (SMAs with specific periods)
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    # Teeth: 8-period SMMA, shifted 5 bars ahead  
    # Lips: 5-period SMMA, shifted 3 bars ahead
    # Using SMA as approximation for SMMA (Williams Alligator uses smoothed moving average)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set initial values to NaN where shift creates invalid data
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Alligator alignment
        bullish_alignment = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        bearish_alignment = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below teeth or bullish alignment breaks
            if close[i] < teeth_aligned[i] or not bullish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above teeth or bearish alignment breaks
            if close[i] > teeth_aligned[i] or not bearish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above teeth with volume surge and bullish alignment
            if (close[i] > teeth_aligned[i] and close[i-1] <= teeth_aligned[i-1] and 
                vol_surge and bullish_alignment):
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below teeth with volume surge and bearish alignment
            elif (close[i] < teeth_aligned[i] and close[i-1] >= teeth_aligned[i-1] and 
                  vol_surge and bearish_alignment):
                position = -1
                signals[i] = -0.25
    
    return signals