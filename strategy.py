#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d Volume Spike
- Williams Alligator (Jaws, Teeth, Lips) on 6h to detect trend direction and alignment
- 1d volume spike (current volume > 2x 20-period median) to confirm institutional participation
- Long: Alligator aligned bullish (Lips > Teeth > Jaws) + volume spike
- Short: Alligator aligned bearish (Lips < Teeth < Jaws) + volume spike
- Exit: Opposite Alligator alignment or loss of volume confirmation
- Works in bull (trend following) and bear (catching sharp moves on volume)
- Target: 15-25 trades/year to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h data for Williams Alligator
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (13,8,5 SMAs with future shifts)
    # Jaws: 13-period SMA shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # Shift 8 bars forward
    
    # Teeth: 8-period SMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # Shift 5 bars forward
    
    # Lips: 5-period SMA shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # Shift 3 bars forward
    
    # 1d volume data for spike detection
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    # 20-period median volume on 1d
    vol_median_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).median()
    vol_spike_1d = vol_1d > (2.0 * vol_median_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaws
        bullish = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        # Bearish alignment: Lips < Teeth < Jaws
        bearish = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        # Volume spike confirmation
        vol_spike = vol_spike_1d_aligned[i] > 0.5
        
        # Long: Bullish alignment + volume spike
        if bullish and vol_spike:
            signals[i] = 0.25
        # Short: Bearish alignment + volume spike
        elif bearish and vol_spike:
            signals[i] = -0.25
        # Exit: Opposite alignment or loss of volume confirmation
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (bearish or not vol_spike)) or
               (signals[i-1] == -0.25 and (bullish or not vol_spike)))):
            signals[i] = 0.0
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WilliamsAlligator_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0