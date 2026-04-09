#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d volume confirmation
# The Alligator identifies trends via three smoothed moving averages.
# Jaw (13-period), Teeth (8-period), Lips (5-period) - all shifted forward.
# In trending markets: Lips > Teeth > Jaw (bull) or Lips < Teeth < Jaw (bear)
# In ranging markets: lines intertwine. Uses 12h timeframe for fewer trades (target: 12-37/year).
# Works in bull/bear by capturing strong trends while avoiding chop via alignment.
# Volume confirmation from 1d ensures institutional participation.

name = "12h_1d_alligator_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12h Williams Alligator (SMMA = smoothed moving average)
    def smma(source, period):
        """Smoothed Moving Average - Williams Alligator uses SMMA"""
        result = np.full_like(source, np.nan)
        if len(source) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Shift forward as per Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # 1d volume confirmation: 20-period average
    vol_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            vol_20[i] = np.mean(df_1d['volume'].iloc[i-19:i+1])
    
    # Align 1d volume to 12h timeframe
    vol_20_12h = align_htf_to_ltf(prices, df_1d, vol_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup for SMMA
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_20_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Lips cross below Teeth (trend weakening) OR insufficient volume
            if lips[i] < teeth[i] or volume[i] < vol_20_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Lips cross above Teeth (trend weakening) OR insufficient volume
            if lips[i] > teeth[i] or volume[i] < vol_20_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter strong trend: Lips > Teeth > Jaw with volume confirmation (bull)
            # OR Lips < Teeth < Jaw with volume confirmation (bear)
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                volume[i] > vol_20_12h[i] * 1.5):
                position = 1
                signals[i] = 0.25
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  volume[i] > vol_20_12h[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals