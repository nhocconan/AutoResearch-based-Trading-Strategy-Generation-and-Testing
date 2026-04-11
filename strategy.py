#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_williams_alligator_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return signals
    
    # Calculate Williams Alligator on daily timeframe
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    # Lips (5-period SMMA, shifted 3 bars)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    
    # Align 1d indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(34, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Long conditions: Lips > Teeth > Jaw (bullish alignment) with volume
        long_signal = volume_confirmed and (lips_val > teeth_val) and (teeth_val > jaw_val)
        
        # Short conditions: Lips < Teeth < Jaw (bearish alignment) with volume
        short_signal = volume_confirmed and (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Exit when alignment breaks (Alligator sleeping)
        exit_long = position == 1 and not ((lips_val > teeth_val) and (teeth_val > jaw_val))
        exit_short = position == -1 and not ((lips_val < teeth_val) and (teeth_val < jaw_val))
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Williams Alligator on 1d timeframe with 12h entries.
# The Alligator uses three SMMA lines (Jaw, Teeth, Lips) to identify trends.
# When Lips > Teeth > Jaw, the Alligator is awake with bullish trend.
# When Lips < Teeth < Jaw, the Alligator is awake with bearish trend.
# When lines are intertwined, the Alligator is sleeping (no trend).
# Volume confirmation ensures participation from market actors.
# This strategy works in both bull and bear markets by following the Alligator's
# alignment. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee
# drag on 12h timeframe. The Alligator's smoothed moving averages reduce noise
# and false signals, while the 1d timeframe provides a higher timeframe trend filter.
# Entry occurs when the Alligator wakes up with clear alignment and volume confirmation.
# Exit when the Alligator goes back to sleep (alignment breaks).
# Expected trades: ~100 total over 4 years to stay within optimal range.