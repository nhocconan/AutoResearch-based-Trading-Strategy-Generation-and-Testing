#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams Alligator with momentum confirmation
# Long when Alligator mouth opens upward (Jaw < Teeth < Lips) + close > Teeth + volume > 1.5x average
# Short when Alligator mouth opens downward (Jaw > Teeth > Lips) + close < Teeth + volume > 1.5x average
# Alligator identifies trend direction and strength. Mouth open indicates strong trend.
# Volume confirms momentum behind the move. Works in bull/bear markets by following established trends.
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing.

name = "4h_1dWilliamsAlligator_Momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Williams Alligator (SMMA) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator uses Smoothed Moving Average (SMMA)
    # Jaw: 13-period SMMA, 8 bars ahead
    # Teeth: 8-period SMMA, 5 bars ahead
    # Lips: 5-period SMMA, 3 bars ahead
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_1d = df_1d['close'].values
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift jaws forward as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align 1-day Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator mouth opens upward (Jaw < Teeth < Lips) + momentum + volume
            if (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and 
                close[i] > teeth_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator mouth opens downward (Jaw > Teeth > Lips) + momentum + volume
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                  close[i] < teeth_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: mouth closes or reverses
            if not (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]) or close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: mouth closes or reverses
            if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals