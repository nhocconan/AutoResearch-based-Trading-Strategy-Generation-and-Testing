#!/usr/bin/env python3
name = "12h_WilliamsAlligator_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_sma(arr, period):
    """Simple moving average with proper NaN handling."""
    sma = np.full_like(arr, np.nan, dtype=np.float64)
    if len(arr) < period:
        return sma
    for i in range(period - 1, len(arr)):
        sma[i] = np.mean(arr[i - period + 1:i + 1])
    return sma

def calculate_williams_alligator(high, low, close):
    """
    Williams Alligator:
    - Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    - Teeth (Red): 8-period SMMA, shifted 5 bars forward
    - Lips (Green): 5-period SMMA, shifted 3 bars forward
    SMMA = Smoothed Moving Average (Wilder's smoothing)
    """
    # Typical price
    typical_price = (high + low + close) / 3.0
    
    # Smoothed Moving Average (SMMA) - Wilder's smoothing
    def smma(arr, period):
        sma = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return sma
        # First value: simple average
        sma[period - 1] = np.mean(arr[0:period])
        # Wilder's smoothing: SMMA[i] = (SMMA[i-1] * (period-1) + price[i]) / period
        for i in range(period, len(arr)):
            sma[i] = (sma[i-1] * (period - 1) + arr[i]) / period
        return sma
    
    jaw_raw = smma(typical_price, 13)
    teeth_raw = smma(typical_price, 8)
    lips_raw = smma(typical_price, 5)
    
    # Shift forward: Jaw +8, Teeth +5, Lips +3
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 1D
    jaw_1d, teeth_1d, lips_1d = calculate_williams_alligator(high_1d, low_1d, close_1d)
    
    # Align 1D indicators to 12H timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Simple volume filter: above average volume
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > volume_sma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data
        # Skip if any Alligator line is NaN
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment signals:
        # Lips > Teeth > Jaw = bullish alignment (green > red > blue)
        # Lips < Teeth < Jaw = bearish alignment (green < red < blue)
        bullish_alignment = (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i])
        bearish_alignment = (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i])
        
        if position == 0:
            # LONG: Bullish alignment + volume confirmation
            if bullish_alignment and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish alignment + volume confirmation
            elif bearish_alignment and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alignment breaks (lips cross below teeth)
            if lips_1d_aligned[i] <= teeth_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alignment breaks (lips cross above teeth)
            if lips_1d_aligned[i] >= teeth_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals