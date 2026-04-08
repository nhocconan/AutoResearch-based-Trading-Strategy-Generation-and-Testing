#!/usr/bin/env python3
# 12h_1d_donchian_volume_v1
# Hypothesis: Trade 12-hour Donchian breakouts with 1-day volume confirmation.
# Enter long when price breaks above 24-period 12h Donchian high with volume > 1.5x 24-period average.
# Enter short when price breaks below 24-period 12h Donchian low with volume > 1.5x 24-period average.
# Exit when price crosses the Donchian midline.
# Target: 12-30 trades/year with strict entry to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (24-period)
    donch_high = pd.Series(high).rolling(window=24, min_periods=24).max().values
    donch_low = pd.Series(low).rolling(window=24, min_periods=24).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume confirmation: 12h volume > 1.5x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 48  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_24[i] if vol_ma_24[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below Donchian midline
            if close[i] < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above Donchian midline
            if close[i] > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Break above Donchian high with volume surge
            if high[i] > donch_high[i-1] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: Break below Donchian low with volume surge
            elif low[i] < donch_low[i-1] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals