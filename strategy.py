#!/usr/bin/env python3
"""
6d_elder_ray_volume_v1
Hypothesis: Elder Ray (Bull/Bear Power) with volume confirmation on 6h timeframe.
- Uses 13-period EMA as trend reference
- Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
- Enter long when Bull Power > 0 and increasing + volume > 1.5x average
- Enter short when Bear Power > 0 and increasing + volume > 1.5x average
- Exit when power decreases or volume dries up
- Works in both bull and bear markets by measuring buying/selling pressure relative to trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6d_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA(13) for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate average volume (50-period)
    volume_series = pd.Series(volume)
    vol_avg = volume_series.rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x average
        vol_condition = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long
            # Exit: Bull Power decreasing or volume dries up
            if bull_power[i] < bull_power[i-1] or not vol_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: Bear Power decreasing or volume dries up
            if bear_power[i] < bear_power[i-1] or not vol_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: Bull Power positive and increasing + volume confirmation
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and vol_condition:
                position = 1
                signals[i] = 0.25
            # Short: Bear Power positive and increasing + volume confirmation
            elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and vol_condition:
                position = -1
                signals[i] = -0.25
    
    return signals