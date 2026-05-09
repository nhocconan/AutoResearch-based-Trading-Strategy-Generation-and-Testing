#!/usr/bin/env python3
# 12h Williams Alligator Trend Strategy
# Uses Williams Alligator (Jaws, Teeth, Lips) on 1d timeframe to determine trend direction
# Enters long when price > Lips in uptrend, short when price < Lips in downtrend
# Uses volume confirmation on 12h to avoid false breakouts
# Target: 20-40 trades/year (~80-160 total over 4 years)
# Works in bull (trend following) and bear (avoids counter-trend via Alligator alignment)

name = "12h_Williams_Alligator_Trend"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for SMAs
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Alligator components (SMAs with specific periods)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA)"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[0:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply forward shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Need volume MA and Alligator components
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend based on Alligator alignment
        # In uptrend: Lips > Teeth > Jaw (all above price typically)
        # In downtrend: Jaw > Teeth > Lips (all below price typically)
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        jaw_above_teeth = jaw_aligned[i] > teeth_aligned[i]
        teeth_above_lips = teeth_aligned[i] > lips_aligned[i]
        
        uptrend = lips_above_teeth and teeth_above_jaw
        downtrend = jaw_above_teeth and teeth_above_lips
        
        volume_surge = volume_ratio[i] > 1.5  # Require 1.5x average volume
        
        if position == 0:
            # Enter long: Uptrend + price above Lips + volume surge
            if uptrend and close[i] > lips_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price below Lips + volume surge
            elif downtrend and close[i] < lips_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend changes to downtrend OR price crosses below Lips
            if not uptrend or close[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend changes to uptrend OR price crosses above Lips
            if not downtrend or close[i] > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals