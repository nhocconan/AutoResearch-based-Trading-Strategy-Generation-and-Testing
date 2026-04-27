#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour strategy using Williams Alligator (SMMA-based) from 1-day timeframe with volume confirmation.
# The Alligator identifies trend direction via jaw/teeth/lips alignment. 
# Long when lips > teeth > jaw (bullish alignment), short when lips < teeth < jaw (bearish alignment).
# Volume confirmation (>1.5x 20-period average) filters weak breakouts.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets (trend following) and bear markets (trend following).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Williams Alligator: three SMMA lines
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = np.zeros_like(series)
        sma[:] = np.nan
        if len(series) >= period:
            sma[period-1] = np.mean(series[:period])
            for i in range(period, len(series)):
                sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    jaw = smma(median_1d, 13)
    teeth = smma(median_1d, 8)
    lips = smma(median_1d, 5)
    
    # Shift jaw forward by 8, teeth by 5, lips by 3 (Alligator rules)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with nan
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Enter long on bullish alignment with volume
        if bullish and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Enter short on bearish alignment with volume
        elif bearish and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit when alignment breaks
        elif position == 1 and not bullish:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not bearish:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeFilter"
timeframe = "12h"
leverage = 1.0