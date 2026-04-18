#!/usr/bin/env python3
"""
4h_WilliamsAlligator_Trend - Williams Alligator (SMMA) trend filter with price action confirmation
Williams Alligator: Jaw (13-period SMMA, 8-shift), Teeth (8-period SMMA, 5-shift), Lips (5-period SMMA, 3-shift)
Trend: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
Entry: Price closes above/below Alligator in trend direction + volume confirmation
Exit: Opposite signal or price crosses middle (Teeth) line
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Works in bull markets (trend following) and bear markets (trend following)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called Wilder's smoothing"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (Prev SMMA * (length-1) + Current Price) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator (more stable on higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components on 1D
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    jaw = smma(median_1d, 13)  # Jaw: 13-period SMMA
    teeth = smma(median_1d, 8)   # Teeth: 8-period SMMA  
    lips = smma(median_1d, 5)    # Lips: 5-period SMMA
    
    # Apply shifts as per Williams Alligator definition
    jaw = np.roll(jaw, 8)   # Jaw shifted 8 bars forward
    teeth = np.roll(teeth, 5) # Teeth shifted 5 bars forward
    lips = np.roll(lips, 3)   # Lips shifted 3 bars forward
    
    # Align to 4H timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation (20-period average on 4H)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for dynamic sizing (optional filter)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator trend detection
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Lips < Teeth < Jaw
        uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        downtrend = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Price position relative to Alligator
        price_above_alligator = close[i] > lips_aligned[i] and close[i] > teeth_aligned[i] and close[i] > jaw_aligned[i]
        price_below_alligator = close[i] < lips_aligned[i] and close[i] < teeth_aligned[i] and close[i] < jaw_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Uptrend + price above Alligator + volume
            if uptrend and price_above_alligator and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + price below Alligator + volume
            elif downtrend and price_below_alligator and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or price crosses below Teeth (middle line)
            if not uptrend or close[i] < teeth_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or price crosses above Teeth (middle line)
            if not downtrend or close[i] > teeth_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_Trend"
timeframe = "4h"
leverage = 1.0