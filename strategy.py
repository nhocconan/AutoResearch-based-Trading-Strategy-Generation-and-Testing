#!/usr/bin/env python3
"""
1d Williams Alligator + Volume Spike + Close Above/Below Teeth
Long: Close > Teeth (red line) + volume > 2.0x 20-day avg volume
Short: Close < Teeth (red line) + volume > 2.0x 20-day avg volume
Exit: Opposite condition
Williams Alligator: Jaw (blue) = SMA(13,8), Teeth (red) = SMA(8,5), Lips (green) = SMA(5,3)
Uses Williams Alligator to identify trend direction, volume spike for confirmation
Target: 15-25 trades/year per symbol
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
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator lines
    # Jaw (blue line): 13-period SMMA shifted 8 bars ahead
    jaw = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth (red line): 8-period SMMA shifted 5 bars ahead
    teeth = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips (green line): 5-period SMMA shifted 3 bars ahead
    lips = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Align to lower timeframe (1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    
    # Volume confirmation: 20-day average volume
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean()
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(teeth_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20_aligned[i]
        teeth_val = teeth_aligned[i]
        
        if position == 0:
            # Long: close above teeth + volume spike
            if price > teeth_val and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: close below teeth + volume spike
            elif price < teeth_val and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below teeth
            if price < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above teeth
            if price > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_Teeth_VolumeSpike"
timeframe = "1d"
leverage = 1.0