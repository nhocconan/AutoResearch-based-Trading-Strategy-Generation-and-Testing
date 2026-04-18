#!/usr/bin/env python3
"""
4h Williams Alligator + Volume Spike + Trend Filter
Hypothesis: Williams Alligator (3 SMAs) identifies trending vs ranging markets. When jaws (13-bar SMA) are open (diverging) and price is outside teeth/lips with volume spike, it signals strong trend continuation. Works in bull/bear by capturing breakouts from consolidation. Low frequency due to strict jaw alignment requirement.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close):
    """Williams Alligator: Jaw(13), Teeth(8), Lips(5) SMAs of median price"""
    median = (high + low) / 2
    jaw = pd.Series(median).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median).rolling(window=5, min_periods=5).mean().values
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    jaw_1d, teeth_1d, lips_1d = calculate_alligator(high_1d, low_1d, close_1d)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume spike: current volume > 2.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        # Check if Alligator is "awake" (jaws open)
        jaws_open = (jaw > teeth and teeth > lips) or (jaw < teeth and teeth < lips)
        
        if position == 0:
            # Enter long: price above all lines + jaws open up + volume spike
            if (close[i] > jaw and close[i] > teeth and close[i] > lips and
                jaw > teeth and teeth > lips and  # jaws open up
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price below all lines + jaws open down + volume spike
            elif (close[i] < jaw and close[i] < teeth and close[i] < lips and
                  jaw < teeth and teeth < lips and  # jaws open down
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below teeth OR jaws close
            if close[i] < teeth or not (jaw > teeth and teeth > lips):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above teeth OR jaws close
            if close[i] > teeth or not (jaw < teeth and teeth < lips):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0