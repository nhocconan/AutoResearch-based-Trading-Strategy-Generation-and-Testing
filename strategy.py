#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with Elder Ray power confirmation and volume spike.
# Long when: Green phase (jaw<teeth<lips) + bull power > 0 + volume spike
# Short when: Red phase (lips<teeth<jaw) + bear power < 0 + volume spike
# Exit when: Opposite Alligator phase or volume drops below 80% of average.
# Uses 1d for Alligator/Elder Ray calculations to reduce noise.
# Target: 20-30 trades/year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Alligator and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator (13,8,5 SMAs with future shifts)
    # Jaw (13-period, 8-bar shift)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period, 5-bar shift)
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period, 3-bar shift)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray Power (13-period EMA)
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        # Alligator phases
        green_phase = (jaw_val < teeth_val) and (teeth_val < lips_val)  # Jaw < Teeth < Lips
        red_phase = (lips_val < teeth_val) and (teeth_val < jaw_val)   # Lips < Teeth < Jaw
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: Green phase + bull power > 0 + volume spike
            if green_phase and bull_power_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Red phase + bear power < 0 + volume spike
            elif red_phase and bear_power_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Opposite phase or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when red phase or volume dries up
                if red_phase or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when green phase or volume dries up
                if green_phase or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0