#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend direction, Elder Ray (Bull/Bear Power) for momentum,
# and volume spike for confirmation. Works in trending markets (bull/bear) by aligning with Alligator's
# teeth/lips crossover and Elder Ray extremes. Volume spike filters low-conviction moves.
# Target: 50-150 total trades over 4 years.

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
    
    # Williams Alligator (13,8,5 SMAs shifted by 8,5,3)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    jaw_shifted = jaw.shift(8)
    teeth_shifted = teeth.shift(5)
    lips_shifted = lips.shift(3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align to 4h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted.values)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power.values)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power.values)
    
    # Volume spike: current volume > 2x median of past 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_spike = volume > (2 * vol_median.values)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i])):
            continue
        
        # Long: Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + volume spike
        if (lips_aligned[i] > teeth_aligned[i] and
            teeth_aligned[i] > jaw_aligned[i] and
            bull_power_aligned[i] > 0 and
            vol_spike[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: Lips < Teeth < Jaw (bearish alignment) + Bear Power < 0 + volume spike
        elif (lips_aligned[i] < teeth_aligned[i] and
              teeth_aligned[i] < jaw_aligned[i] and
              bear_power_aligned[i] < 0 and
              vol_spike[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reversal of Alligator alignment or Elder Power crosses zero
        elif position == 1 and (lips_aligned[i] < teeth_aligned[i] or bull_power_aligned[i] < 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (lips_aligned[i] > teeth_aligned[i] or bear_power_aligned[i] > 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_VolumeSpike"
timeframe = "4h"
leverage = 1.0