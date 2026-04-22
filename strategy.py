#!/usr/bin/env python3

"""
Hypothesis: 6-hour Williams Alligator with 1-week trend filter and volume confirmation.
The Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets. Using weekly
trend filter ensures we only trade in the direction of the higher timeframe trend.
Volume spikes confirm institutional interest. This should work in both bull and bear
markets by adapting to the weekly trend, avoiding counter-trend trades that cause
whipsaw in sideways markets. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator components"""
    # Jaw (Blue): 13-period SMMA shifted 8 bars ahead
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(8)
    
    # Teeth (Red): 8-period SMMA shifted 5 bars ahead
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(5)
    
    # Lips (Green): 5-period SMMA shifted 3 bars ahead
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().shift(3)
    
    return jaw.values, teeth.values, lips.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Alligator for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    jaw_1w, teeth_1w, lips_1w = calculate_alligator(high_1w, low_1w, close_1w)
    
    # Determine weekly trend: Alligator alignment
    # Bullish: Lips > Teeth > Jaw (all aligned upward)
    # Bearish: Lips < Teeth < Jaw (all aligned downward)
    bullish_trend = (lips_1w > teeth_1w) & (teeth_1w > jaw_1w)
    bearish_trend = (lips_1w < teeth_1w) & (teeth_1w < jaw_1w)
    
    # Align Alligator components and trend to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_trend.astype(float))
    
    # Calculate 6h Alligator for entry signals
    jaw_6h, teeth_6h, lips_6h = calculate_alligator(high, low, close)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or np.isnan(jaw_6h[i]) or 
            np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment), bullish weekly trend, volume spike
            if (lips_6h[i] > teeth_6h[i] and 
                teeth_6h[i] > jaw_6h[i] and 
                bullish_aligned[i] > 0.5 and 
                volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment), bearish weekly trend, volume spike
            elif (lips_6h[i] < teeth_6h[i] and 
                  teeth_6h[i] < jaw_6h[i] and 
                  bearish_aligned[i] > 0.5 and 
                  volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator alignment breaks or reverse signal
            exit_signal = False
            
            if position == 1:
                # Exit long: Lips < Teeth OR Teeth < Jaw (bullish alignment broken)
                if (lips_6h[i] < teeth_6h[i] or 
                    teeth_6h[i] < jaw_6h[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Lips > Teeth OR Teeth > Jaw (bearish alignment broken)
                if (lips_6h[i] > teeth_6h[i] or 
                    teeth_6h[i] > jaw_6h[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0