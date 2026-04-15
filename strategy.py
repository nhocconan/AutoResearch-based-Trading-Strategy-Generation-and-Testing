#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 1d volume regime filter
# Uses Alligator to identify trends (JAW=13sma, TEETH=8sma, LIPS=5sma) and trades in direction of alignment
# Volume filter ensures we only trade when 1d volume is above average (avoids low-volume false signals)
# Works in bull/bear by following Alligator alignment: JAW>TEETH>LIPS = uptrend, reverse = downtrend
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data for price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Alligator on 6h
    # Jaw (blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (red): 8-period SMMA, shifted 5 bars forward  
    # Lips (green): 5-period SMMA, shifted 3 bars forward
    # Using SMA as approximation for SMMA (close enough for our purposes)
    jaw_6h = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().values
    teeth_6h = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().values
    lips_6h = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().values
    
    # Shift to align with Alligator methodology
    jaw_6h = np.roll(jaw_6h, 8)
    teeth_6h = np.roll(teeth_6h, 5)
    lips_6h = np.roll(lips_6h, 3)
    # Set shifted values to NaN
    jaw_6h[:8] = np.nan
    teeth_6h[:5] = np.nan
    lips_6h[:3] = np.nan
    
    # Calculate 1d volume average (20-period)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    jaw_6h_aligned = align_htf_to_ltf(prices, df_6h, jaw_6h)
    teeth_6h_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h)
    lips_6h_aligned = align_htf_to_ltf(prices, df_6h, lips_6h)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_6h_aligned[i]) or np.isnan(teeth_6h_aligned[i]) or
            np.isnan(lips_6h_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: Lips > Teeth > Jaw (bullish alignment) + volume above average
        if (lips_6h_aligned[i] > teeth_6h_aligned[i] > jaw_6h_aligned[i] and
            volume[i] > vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Lips < Teeth < Jaw (bearish alignment) + volume above average
        elif (lips_6h_aligned[i] < teeth_6h_aligned[i] < jaw_6h_aligned[i] and
              volume[i] > vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: when Alligator alignment breaks (Lips crosses Teeth)
        elif position == 1 and lips_6h_aligned[i] < teeth_6h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and lips_6h_aligned[i] > teeth_6h_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1dVolume_Filter"
timeframe = "6h"
leverage = 1.0