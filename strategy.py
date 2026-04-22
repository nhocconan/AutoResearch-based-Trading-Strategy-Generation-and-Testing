#!/usr/bin/env python3

"""
Hypothesis: 4-hour Williams Alligator with 1-day trend filter and volume confirmation.
Alligator lines (Jaw/Teeth/Lips) provide trend direction and strength. Using daily trend filter
avoids counter-trend trades. Volume spikes confirm institutional interest.
This should work in both bull and bear regimes by adapting to the daily trend.
Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator lines"""
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
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily Alligator for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    jaw_1d, teeth_1d, lips_1d = calculate_alligator(high_1d, low_1d, close_1d)
    
    # Determine daily trend: Alligator alignment
    # Bullish: Lips > Teeth > Jaw (green > red > blue)
    # Bearish: Jaw > Teeth > Lips (blue > red > green)
    bullish_trend = (lips_1d > teeth_1d) & (teeth_1d > jaw_1d)
    bearish_trend = (jaw_1d > teeth_1d) & (teeth_1d > lips_1d)
    
    # Align Alligator components to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Calculate 4h Alligator for entry signals
    jaw_4h, teeth_4h, lips_4h = calculate_alligator(high, low, close)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(jaw_4h[i]) or np.isnan(teeth_4h[i]) or np.isnan(lips_4h[i]) or 
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
            # Long: Lips > Teeth > Jaw (green > red > blue), bullish daily trend, volume spike
            if (lips_4h[i] > teeth_4h[i] and teeth_4h[i] > jaw_4h[i] and  # Alligator bullish alignment
                bullish_aligned[i] > 0.5 and                             # Bullish daily trend
                volume[i] > 1.8 * vol_avg_20[i]):                        # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (blue > red > green), bearish daily trend, volume spike
            elif (jaw_4h[i] > teeth_4h[i] and teeth_4h[i] > lips_4h[i] and  # Alligator bearish alignment
                  bearish_aligned[i] > 0.5 and                              # Bearish daily trend
                  volume[i] > 1.8 * vol_avg_20[i]):                         # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines cross in opposite direction
            exit_signal = False
            
            if position == 1:
                # Exit long: Lips < Teeth or Teeth < Jaw (loss of bullish alignment)
                if (lips_4h[i] < teeth_4h[i]) or (teeth_4h[i] < jaw_4h[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Jaw < Teeth or Teeth < Lips (loss of bearish alignment)
                if (jaw_4h[i] < teeth_4h[i]) or (teeth_4h[i] < lips_4h[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0