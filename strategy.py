#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator lines from 1d data
    close_1d = df_1d['close'].values
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Convert to numpy arrays
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Align 1d indicators to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume filter on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator lines converge (teeth crosses below lips) OR volume filter fails
            if teeth_aligned[i] < lips_aligned[i] or not volume_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Alligator lines converge (teeth crosses above lips) OR volume filter fails
            if teeth_aligned[i] > lips_aligned[i] or not volume_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Lips > Teeth > Jaw (bullish alignment) + volume
            if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Lips < Teeth < Jaw (bearish alignment) + volume
            elif lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals