#!/usr/bin/env python3
# 6h_fisher_transform_volume_v1
# Hypothesis: Ehlers Fisher Transform identifies turning points in price cycles.
# Works in both bull and bear markets by detecting extreme reversals.
# Fisher > 1.5 = overbought (short), Fisher < -1.5 = oversold (long).
# Volume confirmation filters weak signals.
# Exit on opposite Fisher cross or volume drop.
# Target: 60-120 total trades over 4 years (~15-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_fisher_transform_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Fisher Transform (9-period)
    hl2 = (prices['high'].values + prices['low'].values) / 2
    max_hl2 = pd.Series(hl2).rolling(window=9, min_periods=9).max().values
    min_hl2 = pd.Series(hl2).rolling(window=9, min_periods=9).min().values
    range_hl2 = max_hl2 - min_hl2
    range_hl2 = np.where(range_hl2 == 0, 1e-10, range_hl2)  # avoid division by zero
    
    # Normalize to [-1, 1]
    value = 2 * ((hl2 - min_hl2) / range_hl2) - 1
    value = np.clip(value, -0.999, 0.999)  # prevent log domain issues
    
    # Fisher Transform
    fish = np.zeros(n)
    fish[0] = 0
    for i in range(1, n):
        fish[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * fish[i-1]
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(fish[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Fisher crosses above -1.5 (overbought) or volume drops below average
            if fish[i] > -1.5 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Fisher crosses below 1.5 (oversold) or volume drops below average
            if fish[i] < 1.5 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.3x average volume
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            # Fisher Transform entries: Fisher < -1.5 (long) or > 1.5 (short)
            if fish[i] < -1.5 and volume_ok:
                position = 1
                signals[i] = 0.25
            elif fish[i] > 1.5 and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals