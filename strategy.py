#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with 13/8/5 SMAs (Jaws/Teeth/Lips) and volume confirmation.
# Long when: Lips > Teeth > Jaws (bullish alignment) and volume > 1.5x 20-period average
# Short when: Lips < Teeth < Jaws (bearish alignment) and volume > 1.5x 20-period average
# Exit when: alignment breaks (Lips crosses Teeth) or volume drops below average
# Alligator identifies trend direction and strength; SMAs filter noise; volume confirms conviction.
# Works in bull (buy alignment) and bear (sell alignment). Target: 12-37 trades/year per symbol.
name = "12h_WilliamsAlligator_SMAs_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator SMAs (13, 8, 5 periods)
    jaws = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values    # 8-period SMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values     # 5-period SMA
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        jaw = jaws[i]
        tooth = teeth[i]
        lip = lips[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Bullish alignment (Lips > Teeth > Jaws) and volume spike
            if (lip > tooth > jaw) and (vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment (Lips < Teeth < Jaws) and volume spike
            elif (lip < tooth < jaw) and (vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bullish alignment breaks (Lips crosses below Teeth)
            if lip <= tooth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bearish alignment breaks (Lips crosses above Teeth)
            if lip >= tooth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals