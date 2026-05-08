#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Uses Williams Alligator (3 SMAs: Jaw, Teeth, Lips) to detect trends in both bull/bear markets.
# Requires alignment with weekly EMA50 trend and volume spike to filter false signals.
# Designed for low-frequency trades (<150 total) to minimize fee drag on 12h timeframe.

name = "12h_WilliamsAlligator_1wEMA50_Volume"
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
    
    # Get 1w data for Williams Alligator calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator components (3 SMAs)
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw_1w = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values
    jaw_1w = np.roll(jaw_1w, 8)
    jaw_1w[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth_1w = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values
    teeth_1w = np.roll(teeth_1w, 5)
    teeth_1w[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips_1w = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values
    lips_1w = np.roll(lips_1w, 3)
    lips_1w[:3] = np.nan
    
    # Align Williams Alligator components to 12h timeframe
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Get 1w data for EMA50 trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike (2.0x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 has enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or 
            np.isnan(lips_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals: Lips above Teeth above Jaw = uptrend
        # Lips below Teeth below Jaw = downtrend
        lips_above_teeth = lips_1w_aligned[i] > teeth_1w_aligned[i]
        teeth_above_jaw = teeth_1w_aligned[i] > jaw_1w_aligned[i]
        lips_below_teeth = lips_1w_aligned[i] < teeth_1w_aligned[i]
        teeth_below_jaw = teeth_1w_aligned[i] < jaw_1w_aligned[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw with 1w uptrend and volume spike
            if (lips_above_teeth and teeth_above_jaw and 
                close[i] > ema50_1w_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw with 1w downtrend and volume spike
            elif (lips_below_teeth and teeth_below_jaw and 
                  close[i] < ema50_1w_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator reverses (Lips < Teeth) or trend fails
            if (lips_1w_aligned[i] < teeth_1w_aligned[i] or 
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator reverses (Lips > Teeth) or trend fails
            if (lips_1w_aligned[i] > teeth_1w_aligned[i] or 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals