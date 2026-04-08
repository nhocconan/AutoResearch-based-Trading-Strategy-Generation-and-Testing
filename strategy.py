#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray + Volume Filter
Hypothesis: The Williams Alligator defines market structure (sleeping/awake/feeding), 
while Elder Ray measures bull/bear power. Combining these with volume confirmation 
creates a robust trend-following system that works in both bull and bear markets 
by only taking trades when all three indicators align. The 6h timeframe targets 
12-37 trades/year, avoiding excessive turnover while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_alligator_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator (13,8,5) - Smoothed Medians
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator sleeping (jaws below teeth) or bear power dominates
            if jaw[i] < teeth[i] or bear_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator sleeping (jaws above teeth) or bull power dominates
            if jaw[i] > teeth[i] or bull_power[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Alligator awake: lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
            # Long: bullish alignment + bull power positive + volume
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                bull_power[i] > 0 and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: bearish alignment + bear power negative + volume
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  bear_power[i] < 0 and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals