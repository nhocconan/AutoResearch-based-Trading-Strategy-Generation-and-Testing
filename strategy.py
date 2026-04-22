#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator trend + 1d volume spike + 12h price position
    # Williams Alligator (Jaw/Teeth/Lips) identifies strong trends; trades only in aligned direction
    # Volume spike confirms institutional participation; avoids chop/false breakouts
    # Works in bull/bear: Alligator adapts to trend direction; volume filters noise
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Williams Alligator: SMAs of median price (hlc3)
    hlc3_12h = (df_12h['high'].values + df_12h['low'].values + close_12h) / 3
    jaw = pd.Series(hlc3_12h).rolling(window=13, min_periods=13).mean().values  # Blue line (13-period)
    teeth = pd.Series(hlc3_12h).rolling(window=8, min_periods=8).mean().values   # Red line (8-period)
    lips = pd.Series(hlc3_12h).rolling(window=5, min_periods=5).mean().values    # Green line (5-period)
    
    # Shift for proper alignment (Williams Alligator uses future-shifted SMAs)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data for volume spike
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > 2.0 * vol_ma20_1d  # Volume spike: >2x 20-period MA
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume spike
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume spike
            elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: When Alligator lines intertwine (trend weakening) OR volume drops
            if position == 1:
                if not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]) or not vol_spike_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]) or not vol_spike_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0