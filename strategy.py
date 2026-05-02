#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Elder Ray + volume confirmation
# Uses 12h timeframe for primary signal generation with Williams Alligator (JAW/TEETH/LIPS) for trend direction
# 1d Elder Ray (Bull/Bear Power) as additional trend confirmation filter
# Volume spike (2.0x 20-period average) ensures institutional participation
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag on 12h timeframe
# Williams Alligator catches trends early, Elder Ray filters false signals, volume confirms strength
# Works in bull markets via Alligator alignment (Lips > Teeth > Jaw) and in bear via reverse alignment
# Session filter (08-20 UTC) reduces noise outside active hours

name = "12h_WilliamsAlligator_1dElderRay_VolumeConfirm_v1"
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
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    median_price_12h = (high_12h + low_12h) / 2.0
    
    # Jaw (Blue): 13-period SMMA, shifted 8 bars ahead
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth (Red): 8-period SMMA, shifted 5 bars ahead
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips (Green): 5-period SMMA, shifted 3 bars ahead
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Align Alligator components to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw.values)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth.values)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips.values)
    
    # Load 1d data ONCE before loop for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray (Bull Power/Bear Power)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray components to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation (2.0x 20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Lips > Teeth > Jaw (Alligator aligned up) + Bull Power > 0 + volume confirm
            if (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i] and 
                bull_power_aligned[i] > 0 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (Alligator aligned down) + Bear Power < 0 + volume confirm
            elif (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i] and 
                  bear_power_aligned[i] < 0 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment breaks (Lips < Teeth OR Teeth < Jaw) OR Bear Power > 0
            if (lips_12h_aligned[i] < teeth_12h_aligned[i] or 
                teeth_12h_aligned[i] < jaw_12h_aligned[i] or 
                bear_power_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (Lips > Teeth OR Teeth > Jaw) OR Bull Power < 0
            if (lips_12h_aligned[i] > teeth_12h_aligned[i] or 
                teeth_12h_aligned[i] > jaw_12h_aligned[i] or 
                bull_power_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals