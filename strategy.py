#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d Elder Ray trend filter and volume confirmation
- Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
- 1d Elder Ray (Bull/Bear Power) confirms higher timeframe trend strength
- Volume spike (> 2.0x 20-period average) filters false signals
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d Elder Ray trend
- Alligator provides dynamic support/resistance with proven edge in ranging markets
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
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d Elder Ray (Bull Power and Bear Power)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA
    bear_power_1d = low_1d - ema_13_1d   # Bear Power = Low - EMA
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(21, 20)  # for Alligator and volume MA
    
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
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + volume
            bullish_alignment = (lips_aligned[i] > teeth_aligned[i] and 
                               teeth_aligned[i] > jaw_aligned[i])
            long_signal = bullish_alignment and (bull_power_aligned[i] > 0) and (volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) + Bear Power < 0 + volume
            bearish_alignment = (lips_aligned[i] < teeth_aligned[i] and 
                               teeth_aligned[i] < jaw_aligned[i])
            short_signal = bearish_alignment and (bear_power_aligned[i] < 0) and (volume[i] > 2.0 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator reverses or Elder Ray diverges
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish alignment or Bull Power turns negative
                bearish_alignment = (lips_aligned[i] < teeth_aligned[i] and 
                                   teeth_aligned[i] < jaw_aligned[i])
                if bearish_alignment or (bull_power_aligned[i] <= 0):
                    exit_signal = True
            elif position == -1:
                # Exit short: bullish alignment or Bear Power turns positive
                bullish_alignment = (lips_aligned[i] > teeth_aligned[i] and 
                                 teeth_aligned[i] > jaw_aligned[i])
                if bullish_alignment or (bear_power_aligned[i] >= 0):
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