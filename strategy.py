#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Confirmation
# Uses Williams Alligator (3 SMAs) for trend direction, Elder Ray (bull/bear power) for momentum,
# and volume spike for confirmation. Works in trending markets (both bull and bear).
# Target: 50-150 total trades over 4 years (12-37/year).
# Timeframe: 12h, HTF: 1d for Williams Alligator/Elder Ray

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align indicators to 12h timeframe with proper delay
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            continue
        
        # Long entry: Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + Volume spike
        if (lips_aligned[i] > teeth_aligned[i] and 
            teeth_aligned[i] > jaw_aligned[i] and
            bull_power_aligned[i] > 0 and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Jaw > Teeth > Lips (bearish alignment) + Bear Power < 0 + Volume spike
        elif (jaw_aligned[i] > teeth_aligned[i] and 
              teeth_aligned[i] > lips_aligned[i] and
              bear_power_aligned[i] < 0 and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Alligator lines cross or power signals weaken
        elif position == 1 and (lips_aligned[i] <= teeth_aligned[i] or bull_power_aligned[i] <= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (jaw_aligned[i] <= teeth_aligned[i] or bear_power_aligned[i] >= 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Williams_Alligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0