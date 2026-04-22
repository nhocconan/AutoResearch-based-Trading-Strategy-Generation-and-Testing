#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator (3 SMAs) with 1d volume confirmation
    # Alligator identifies trend: Jaw (13), Teeth (8), Lips (5) SMAs
    # In uptrend: Lips > Teeth > Jaw; Downtrend: Lips < Teeth < Jaw
    # Volume confirmation filters false signals
    # Works in bull/bear: catches trends early with confirmation
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for Alligator (same as primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Alligator lines (SMAs)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(close_12h).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(close_12h).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close_12h).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Volume MA (20-period) for confirmation
    volume_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(volume_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume above average
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                volume_ma20_aligned[i] > 0 and 
                volume[i] > volume_ma20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume above average
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  volume_ma20_aligned[i] > 0 and 
                  volume[i] > volume_ma20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: When Alligator lines converge (Lips crosses Teeth)
            if position == 1:
                if lips_aligned[i] <= teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips_aligned[i] >= teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_LipsTeethJaw_1dVolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0