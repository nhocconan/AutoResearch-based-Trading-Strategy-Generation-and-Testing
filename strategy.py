#!/usr/bin/env python3
"""
4h Williams Alligator with Volume and Volume Oscillator Filter
Long: Price above Alligator teeth (SMA13) AND Alligator lines aligned bullish (jaw<teeth<lips) AND volume > 1.5x volume SMA(20) AND volume oscillator positive
Short: Price below Alligator teeth AND Alligator lines aligned bearish (jaw>teeth>lips) AND volume > 1.5x volume SMA(20) AND volume oscillator negative
Exit: Price crosses back below/above teeth or Alligator alignment breaks
Uses Williams Alligator for trend/filter, volume for confirmation, volume oscillator for momentum
Target: 20-50 trades/year per symbol (80-200 total over 4 years)
"""

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
    
    # Get 4h data for Williams Alligator
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Williams Alligator on 4h: Jaw (SMA13), Teeth (SMA8), Lips (SMA5)
    jaw_4h = pd.Series(df_4h['close'].values).rolling(window=13, min_periods=13).mean().values
    teeth_4h = pd.Series(df_4h['close'].values).rolling(window=8, min_periods=8).mean().values
    lips_4h = pd.Series(df_4h['close'].values).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 1h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw_4h)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth_4h)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips_4h)
    
    # Calculate volume oscillator on 1h: (fast vol SMA - slow vol SMA) / slow vol SMA
    vol_fast = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    vol_slow = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_osc = (vol_fast - vol_slow) / vol_slow  # positive = increasing volume momentum
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 13)  # Need enough data for slow vol SMA and jaw
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_osc[i]) or np.isnan(vol_slow[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_slow_val = vol_slow[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        osc_val = vol_osc[i]
        
        if position == 0:
            # Long: Price above teeth AND bullish alignment (jaw < teeth < lips) AND volume > 1.5x slow vol SMA AND vol osc positive
            if price > teeth_val and jaw_val < teeth_val and teeth_val < lips_val and vol > 1.5 * vol_slow_val and osc_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: Price below teeth AND bearish alignment (jaw > teeth > lips) AND volume > 1.5x slow vol SMA AND vol osc negative
            elif price < teeth_val and jaw_val > teeth_val and teeth_val > lips_val and vol > 1.5 * vol_slow_val and osc_val < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below teeth OR bullish alignment breaks
            if price < teeth_val or not (jaw_val < teeth_val and teeth_val < lips_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above teeth OR bearish alignment breaks
            if price > teeth_val or not (jaw_val > teeth_val and teeth_val > lips_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_VolumeOsc"
timeframe = "4h"
leverage = 1.0