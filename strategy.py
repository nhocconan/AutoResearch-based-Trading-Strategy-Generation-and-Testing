#!/usr/bin/env python3
"""
12h Williams Alligator with 1d Trend Filter and Volume Confirmation.
Long when price > Alligator's Jaw in 1d uptrend with volume confirmation.
Short when price < Alligator's Jaw in 1d downtrend with volume confirmation.
Exit when price crosses back below/above Jaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_alligator_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D EMA TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)
    
    # === WILLIAMS ALLIGATOR (12H) ===
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # First 8 values of jaw_shifted are invalid
    jaw_shifted[:8] = np.nan
    # First 5 values of teeth_shifted are invalid
    teeth_shifted[:5] = np.nan
    # First 3 values of lips_shifted are invalid
    lips_shifted[:3] = np.nan
    
    # === VOLUME CONFIRMATION (12H) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(one_d_ema_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        uptrend = close[i] > one_d_ema_aligned[i]
        downtrend = close[i] < one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Jaw OR trend turns down
            if close[i] < jaw_shifted[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Jaw OR trend turns up
            if close[i] > jaw_shifted[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Alligator alignment with trend
            # Mouth open upwards (Lips > Teeth > Jaw) in uptrend -> long
            # Mouth open downwards (Lips < Teeth < Jaw) in downtrend -> short
            if (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]) and uptrend:
                # Mouth open up in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]) and downtrend:
                # Mouth open down in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals