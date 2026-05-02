#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray + volume confirmation
# Williams Alligator (jaw/teeth/lips) defines trend structure on 6h timeframe
# 1d Elder Ray (Bull Power/Bear Power) confirms institutional momentum direction
# Volume spike (2.0x 20-period average) filters for high-conviction moves
# Designed to work in bull markets (Alligator aligned up + Bull Power > 0)
# and bear markets (Alligator aligned down + Bear Power < 0)
# Targets 12-37 trades/year (50-150 total over 4 years) for 6h timeframe

name = "6h_WilliamsAlligator_1dElderRay_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data ONCE before loop for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h: Jaw(13,8), Teeth(8,5), Lips(5,3)
    close_6h = df_6h['close'].values
    jaw = pd.Series(close_6h).ewm(span=13, adjust=False).mean().shift(8).values
    teeth = pd.Series(close_6h).ewm(span=8, adjust=False).mean().shift(5).values
    lips = pd.Series(close_6h).ewm(span=5, adjust=False).mean().shift(3).values
    
    # Align Alligator lines to 6h primary timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Load 1d data ONCE before loop for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Align Elder Ray to 6h primary timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate volume spike (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator aligned up: Lips > Teeth > Jaw
            alligator_up = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
            # Alligator aligned down: Jaw > Teeth > Lips
            alligator_down = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
            
            # Long: Alligator up + Bull Power > 0 + volume spike
            if alligator_up and bull_power_aligned[i] > 0 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator down + Bear Power < 0 + volume spike
            elif alligator_down and bear_power_aligned[i] < 0 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment breaks down OR Bull Power turns negative
            if not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]) or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks up OR Bear Power turns positive
            if not (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]) or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals