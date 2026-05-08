#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Uses Williams Alligator (Jaw, Teeth, Lips) on 12h to identify trend direction.
# Requires alignment with 1d EMA50 trend and volume spike to filter false signals.
# Designed for low-frequency trades (target: 50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the trend with proper filters.

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw = pd.Series(close_12h).rolling(window=13, center=False).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth = pd.Series(close_12h).rolling(window=8, center=False).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips (5-period SMMA, shifted 3 bars)
    lips = pd.Series(close_12h).rolling(window=5, center=False).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe (already aligned via get_htf_data)
    # But we need to align to the original 12h timeframe for use in 12h strategy
    jaw_aligned = jaw  # Already in 12h timeframe
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike (2x 20-period EMA on 12h)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + 1d uptrend + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + 1d downtrend + volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks or trend fails
            if (lips_aligned[i] <= teeth_aligned[i] or 
                teeth_aligned[i] <= jaw_aligned[i] or
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks or trend fails
            if (lips_aligned[i] >= teeth_aligned[i] or 
                teeth_aligned[i] >= jaw_aligned[i] or
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals