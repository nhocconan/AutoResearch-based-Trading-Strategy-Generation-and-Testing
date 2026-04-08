#!/usr/bin/env python3
# 12h_1d_cam_pivot_volume_v2
# Hypothesis: Camarilla pivot levels on 1d combined with volume surge and close position.
# Long when price closes above Camarilla H4 level with volume surge.
# Short when price closes below Camarilla L4 level with volume surge.
# Exit when price crosses back below/above H4/L4 or volume drops below average.
# Uses daily pivot levels for structure, volume for confirmation, aims for 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_cam_pivot_volume_v2"
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # Actually standard Camarilla: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # Where C = (H+L+C)/3? No, Camarilla uses previous close as base
    # Standard: H4 = close + (high - low) * 1.1 / 2
    #          L4 = close - (high - low) * 1.1 / 2
    hl_range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + hl_range_1d * 1.1 / 2
    camarilla_l4 = close_1d - hl_range_1d * 1.1 / 2
    
    # Align to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below H4 or volume drops
            if close[i] < camarilla_h4_aligned[i] or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above L4 or volume drops
            if close[i] > camarilla_l4_aligned[i] or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above H4 with volume surge
            if close[i] > camarilla_h4_aligned[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below L4 with volume surge
            elif close[i] < camarilla_l4_aligned[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals