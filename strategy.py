#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and strength.
# Requires alignment with 1d EMA50 trend and volume spike to filter false signals.
# Designed for low-frequency trades (50-150 total) to minimize fee drift.
# Williams Alligator works in both trending and ranging markets by showing convergence/divergence.

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h data
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Close) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply forward shifts as per Williams Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Fill shifted values with NaN where needed
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike (2.0x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals:
        # Mouth open (Lips > Teeth > Jaw) = uptrend
        # Mouth open reversed (Lips < Teeth < Jaw) = downtrend
        # Mouth closed (all intertwined) = ranging/no trend
        
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        
        if position == 0:
            # Enter long: Alligator mouth open up + 1d uptrend + volume spike
            if (lips_val > teeth_val > jaw_val and 
                close[i] > ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator mouth open down + 1d downtrend + volume spike
            elif (lips_val < teeth_val < jaw_val and 
                  close[i] < ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator mouth closes or reverses down
            if not (lips_val > teeth_val > jaw_val):  # Mouth not open up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator mouth closes or reverses up
            if not (lips_val < teeth_val < jaw_val):  # Mouth not open down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals