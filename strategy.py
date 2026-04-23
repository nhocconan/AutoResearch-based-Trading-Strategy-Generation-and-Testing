#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d Elder Ray trend filter and volume confirmation
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) defines 12h trend: Jaw > Teeth > Lips = uptrend, reverse = downtrend
- 1d Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low confirms trend strength
- Volume confirmation (> 1.5x 20-period average) filters false signals
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in bull markets (Alligator uptrend + Bull Power > 0) and bear markets (Alligator downtrend + Bear Power > 0)
- Elder Ray adds institutional trend confirmation beyond simple moving averages
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator uptrend + Bull Power > 0 + volume
            alligator_long = (jaw_aligned[i] > teeth_aligned[i] and 
                            teeth_aligned[i] > lips_aligned[i])
            elder_long = bull_power_aligned[i] > 0
            volume_ok = volume[i] > 1.5 * vol_ma[i]
            
            # Short conditions: Alligator downtrend + Bear Power > 0 + volume
            alligator_short = (jaw_aligned[i] < teeth_aligned[i] and 
                             teeth_aligned[i] < lips_aligned[i])
            elder_short = bear_power_aligned[i] > 0
            
            if alligator_long and elder_long and volume_ok:
                signals[i] = 0.25
                position = 1
            elif alligator_short and elder_short and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator reverses or Elder Ray weakens
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator turns downtrend or Bull Power <= 0
                alligator_down = (jaw_aligned[i] < teeth_aligned[i] and 
                                teeth_aligned[i] < lips_aligned[i])
                elder_weak = bull_power_aligned[i] <= 0
                if alligator_down or elder_weak:
                    exit_signal = True
            elif position == -1:
                # Exit short: Alligator turns uptrend or Bear Power <= 0
                alligator_up = (jaw_aligned[i] > teeth_aligned[i] and 
                              teeth_aligned[i] > lips_aligned[i])
                elder_weak = bear_power_aligned[i] <= 0
                if alligator_up or elder_weak:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dElderRay_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0