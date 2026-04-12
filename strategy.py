#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_alligator_elder_ray_v1
# Combines Williams Alligator (trend detection) with Elder Ray (Bull/Bear Power) on 1h timeframe.
# Uses 1d timeframe to calculate Alligator (Jaw, Teeth, Lips) and Elder Ray components.
# Long when: price > Teeth AND Bull Power > 0 AND Bear Power < 0 (strong uptrend)
# Short when: price < Teeth AND Bear Power > 0 AND Bull Power < 0 (strong downtrend)
# Uses volume confirmation: volume > 1.5 * 50-period average
# Designed for low trade frequency (target: 12-37/year on 6h) to minimize fee drag.
# Works in bull markets (rides uptrends) and bear markets (rides downtrends) by following the Alligator's alignment.

name = "6h_1d_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator parameters (13, 8, 5) with future shifts
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8   # future shift
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate SMAs
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(close_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Apply future shifts (Alligator looks into future)
    jaw = np.roll(jaw, -jaw_shift)
    teeth = np.roll(teeth, -teeth_shift)
    lips = np.roll(lips, -lips_shift)
    
    # Fill NaN from rolling and shifts
    jaw = np.where(np.isnan(jaw), close_1d, jaw)
    teeth = np.where(np.isnan(teeth), close_1d, teeth)
    lips = np.where(np.isnan(lips), close_1d, lips)
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13
    bear_power = df_1d['low'].values - ema13
    
    # Align all indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: volume > 1.5 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if any values not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if no volume
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: price > Teeth AND Bull Power > 0 AND Bear Power < 0
        # (Teeth is the middle line, represents the trend)
        if (close[i] > teeth_aligned[i] and 
            bull_power_aligned[i] > 0 and 
            bear_power_aligned[i] < 0 and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short conditions: price < Teeth AND Bear Power > 0 AND Bull Power < 0
        elif (close[i] < teeth_aligned[i] and 
              bear_power_aligned[i] > 0 and 
              bull_power_aligned[i] < 0 and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: when the Alligator wakes up (Lips crosses Jaw) OR opposite signal
        elif ((lips_aligned[i] > jaw_aligned[i] and position == -1) or  # Lips above Jaw = potential uptrend, exit short
              (lips_aligned[i] < jaw_aligned[i] and position == 1)):   # Lips below Jaw = potential downtrend, exit long
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals